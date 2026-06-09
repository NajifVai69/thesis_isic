"""Lightweight hybrid CNN-ViT with clinical metadata cross-attention fusion.

The proposed model for ISIC-2019. Pipeline:

    Image (3 × 224 × 224)
        │
        ▼
    CNN stem  ─── truncated MobileNetV2 (stages 0–3), ImageNet-pretrained
        │       outputs (B, 96, 14, 14) — local dermoscopic texture
        ▼
    1×1 projection 96 → d_model  +  [CLS] token  +  2D positional embed
        │       → (B, 197, d_model) sequence
        ▼
    ViT trunk ─── N pre-norm transformer encoder blocks (self-attention)
        │       outputs (B, 197, d_model) — global structure (asymmetry, border)
        ▼
    take CLS token → (B, 1, d_model)
        │
        ▼   (optional, when use_metadata=True)
    Cross-attention head ─── CLS as Query attends over metadata tokens
        │       Q = CLS,  K,V = embed({age, sex, anatom_site})  (3 tokens)
        │       missing-value handling: learned `-1` token per field
        ▼
    Linear classifier → 8 logits

Budget (final design — verified via verify_budget + fvcore):
    CNN stem (timm MNv2 features_only) : ~1.81 M params,  ~0.44 GMAC
      NOTE: timm features_only loads all backbone weights including
      blocks not used in the forward pass; actual compute is only
      to the stride-16 stage but all ~1.81M params are in memory.
    2×2 stride-2 patch proj 14→7      :   ~74 K params,  ~0.004 GMAC
    CLS token + positional embedding   :   ~10 K params,  negligible
    ViT trunk (6 × dim 192, mlp×2)    : ~1.78 M params,  ~0.09 GMAC
    Metadata + cross-attention         : ~0.30 M params,  ~0.001 GMAC
    Classifier head                    :  ~1.5 K params,  negligible
    ────────────────────────────────────────────────────────────────
    Total                              : ~3.98 M params,  ~0.63 GMAC

Well inside the < 6 M params / < 1 GMAC target.

Ablation variants (handled by `create_hybrid`):
    hybrid_full       — CNN + ViT + metadata cross-attention  (proposed)
    hybrid_no_meta    — CNN + ViT, classifier on CLS          (ablation: meta)
    hybrid_cnn_only   — CNN stem + GAP + linear               (ablation: ViT)
    hybrid_vit_only   — pure ViT-Tiny on 16×16 patches        (ablation: CNN)

References
    Sandler et al.   "MobileNetV2: Inverted Residuals…"        CVPR 2018
    Dosovitskiy +al. "An Image is Worth 16×16 Words"           ICLR 2021
    Mehta & Rastegari "MobileViT"                              ICLR 2022
    Pacheco & Krohling "An attention-based mechanism…"         Comp.Biol.Med 2021
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── CNN stem (MobileNetV2 backbone, ImageNet-pretrained) ──────────────────────

class CNNStem(nn.Module):
    """Truncated MobileNetV2 producing a 14×14 feature map.

    Uses timm's `mobilenetv2_100` (ImageNet-1k pretrained) in features-only
    mode and extracts at the H/16 stage (out_indices=(3,)), giving a
    `(B, 96, 14, 14)` feature map that becomes the ViT patch grid.

    Why MobileNetV2 specifically:
      - Inverted residuals are well-matched to the small-lesion-on-large-
        background structure typical of dermoscopy.
      - ImageNet-1k pretraining transfers well; the early conv filters
        encode generic edges/textures that dermoscopic images share.
      - 625 K params for stages 0–3 leaves a generous budget for ViT.
    """

    def __init__(self, pretrained: bool = True, out_stage: int = 3):
        super().__init__()
        # Lazy import — keeps `models/hybrid.py` importable without timm
        # in test/CI contexts.
        import timm
        self.backbone = timm.create_model(
            "mobilenetv2_100",
            pretrained=pretrained,
            features_only=True,
            out_indices=(out_stage,),
        )
        self.out_channels = self.backbone.feature_info.channels()[-1]
        self.reduction    = self.backbone.feature_info.reduction()[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)[-1]                       # (B, C, H/16, W/16)


# ── Stochastic depth ──────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Per-sample stochastic depth (Huang et al., 2016).

    Drops the entire residual branch with probability `drop_prob` during
    training. At eval time it is a no-op. This is the standard ViT-era
    regularisation; without it, small ViTs trained from scratch on ~17 K
    images overfit hard.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)        # broadcast over tokens / dims
        mask  = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


# ── Attention (self + cross, with fused SDPA) ─────────────────────────────────

class MultiheadAttention(nn.Module):
    """Multi-head attention that supports both self- and cross-attention.

    When `kv` is None it runs self-attention on `q`; otherwise it cross-
    attends q to the provided key/value source (used by the metadata head).
    Uses `F.scaled_dot_product_attention` so PyTorch picks the fastest
    kernel available (FlashAttention on Ada Lovelace, memory-efficient
    fallback elsewhere).
    """

    def __init__(self, dim: int, num_heads: int = 4,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.attn_drop_p = float(attn_drop)

        self.q_proj  = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor,
                kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        if kv is None:
            kv = q
        B, Nq, D = q.shape
        Nk = kv.shape[1]

        # (B, Nq, D) -> (B, H, Nq, head_dim)
        q_ = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        kv_ = self.kv_proj(kv).reshape(B, Nk, 2, self.num_heads, self.head_dim)
        k_, v_ = kv_.permute(2, 0, 3, 1, 4)                # each: (B, H, Nk, head_dim)

        out = F.scaled_dot_product_attention(
            q_, k_, v_,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, Nq, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


# ── Transformer blocks (self-attention and cross-attention) ───────────────────

class Mlp(nn.Module):
    """Standard transformer feed-forward (GELU, 2-layer)."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1   = nn.Linear(dim, hidden)
        self.act   = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2   = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm self-attention block (standard ViT architecture)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1     = nn.LayerNorm(dim)
        self.attn      = MultiheadAttention(dim, num_heads, attn_drop, drop)
        self.drop_path = DropPath(drop_path)
        self.norm2     = nn.LayerNorm(dim)
        self.mlp       = Mlp(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttentionBlock(nn.Module):
    """One cross-attention block: query attends to an external key/value source.

    Used as the metadata-fusion head: query = CLS token, key/value = 3
    metadata tokens (age, sex, anatomical site). Both inputs are layer-
    normalised before attention (pre-LN), matching the ViT convention.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn    = MultiheadAttention(dim, num_heads, attn_drop, drop)
        self.norm2   = nn.LayerNorm(dim)
        self.mlp     = Mlp(dim, mlp_ratio, drop)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = q + self.attn(self.norm_q(q), self.norm_kv(kv))
        q = q + self.mlp(self.norm2(q))
        return q


# ── Metadata embedding (handles missing values explicitly) ────────────────────

class MetadataEmbedding(nn.Module):
    """Embed clinical metadata (age, sex, anatomical site) → 3 tokens.

    The dataset provides:
        age          (float, z-scored; NaN replaced with 0 by ISICDataset)
        age_missing  (0 or 1)
        sex_idx      (-1 missing | 0 male | 1 female)
        site_idx     (-1 missing | 0..7 site index)

    Missing-value strategy (matches what Pacheco & Krohling 2021 argue is
    superior to imputation): every field has a dedicated learned vector
    that is substituted when the value is missing. The model sees an
    explicit "I don't know" signal instead of a fabricated number.

    Outputs a (B, 3, dim) sequence: [age_token, sex_token, site_token].
    """

    NUM_SEX  = 3        # 0 male, 1 female, 2 missing
    NUM_SITE = 9        # 0..7 sites, 8 missing

    def __init__(self, dim: int):
        super().__init__()
        # Age is continuous → linear projection from scalar; separately
        # we keep a learned "missing" vector that replaces the projection
        # output when age_missing == 1.
        self.age_proj          = nn.Linear(1, dim)
        self.age_missing_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Sex and site are categorical → embeddings with an extra "missing"
        # slot at the last index.
        self.sex_embed  = nn.Embedding(self.NUM_SEX,  dim)
        self.site_embed = nn.Embedding(self.NUM_SITE, dim)

        # Per-token type embedding so the cross-attention can tell the
        # 3 tokens apart even after the values are mixed.
        self.type_embed = nn.Parameter(torch.zeros(1, 3, dim))

        nn.init.trunc_normal_(self.age_missing_token, std=0.02)
        nn.init.trunc_normal_(self.sex_embed.weight,  std=0.02)
        nn.init.trunc_normal_(self.site_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.type_embed,        std=0.02)

    def forward(self, meta: dict) -> torch.Tensor:
        """meta keys: 'age' (B,), 'age_missing' (B,), 'sex_idx' (B,), 'site_idx' (B,)."""
        age         = meta["age"]
        age_missing = meta["age_missing"]
        sex_idx     = meta["sex_idx"]
        site_idx    = meta["site_idx"]
        B = age.shape[0]

        # Age: linear projection, swap in missing token where flagged
        age_tok = self.age_proj(age.float().unsqueeze(-1)).unsqueeze(1)   # (B, 1, dim)
        miss    = age_missing.view(B, 1, 1).bool()
        age_tok = torch.where(miss, self.age_missing_token.expand(B, 1, -1), age_tok)

        # Sex / site: remap -1 (missing) into the last vocab slot
        sex_idx  = sex_idx.long().clone()
        sex_idx[sex_idx  == -1] = self.NUM_SEX  - 1
        site_idx = site_idx.long().clone()
        site_idx[site_idx == -1] = self.NUM_SITE - 1

        sex_tok  = self.sex_embed(sex_idx).unsqueeze(1)                   # (B, 1, dim)
        site_tok = self.site_embed(site_idx).unsqueeze(1)                 # (B, 1, dim)

        tokens = torch.cat([age_tok, sex_tok, site_tok], dim=1)           # (B, 3, dim)
        return tokens + self.type_embed                                   # type-tagged


# ── Hybrid CNN–ViT (the proposed model) ───────────────────────────────────────

class HybridCNNViT(nn.Module):
    """Lightweight hybrid CNN–ViT with optional metadata cross-attention.

    The shared backbone for three of the four ablation variants:

        use_vit=True  + use_metadata=True   → 'hybrid_full'      (proposed)
        use_vit=True  + use_metadata=False  → 'hybrid_no_meta'
        use_vit=False + use_metadata=False  → 'hybrid_cnn_only'

    The fourth (`hybrid_vit_only`) is the separate `PureViT` class.
    """

    def __init__(
        self,
        num_classes:    int   = 8,
        dim:            int   = 192,
        depth:          int   = 6,
        num_heads:      int   = 4,
        mlp_ratio:      float = 2.0,
        drop_rate:      float = 0.0,
        drop_path_rate: float = 0.1,
        use_vit:        bool  = True,
        use_metadata:   bool  = True,
        cnn_pretrained: bool  = True,
    ):
        super().__init__()
        self.use_vit      = use_vit
        self.use_metadata = use_metadata
        self.dim          = dim

        # ── CNN stem (always present) ───────────────────────────────────────
        self.stem = CNNStem(pretrained=cnn_pretrained, out_stage=3)
        feat_ch  = self.stem.out_channels                                  # 96 for MNv2-100

        if use_vit:
            # ── Patch projection: 14×14×C → 49 tokens of dim `dim` ──────
            # kernel=2, stride=2 halves the spatial dims (14→7) while also
            # serving as a learned patch embedding.  Using 49 instead of 196
            # tokens reduces attention cost by ~16× (O(n²)) while the CNN
            # already encoded local texture at full 14×14 resolution.
            self.proj = nn.Conv2d(feat_ch, dim, kernel_size=2, stride=2, bias=True)

            # CLS + learnable positional embedding
            self.num_patches = 7 * 7                                       # for 224×224 input
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

            # Transformer blocks with linearly-increasing stochastic depth
            dpr = [x.item() for x in torch.linspace(0.0, drop_path_rate, depth)]
            self.blocks = nn.ModuleList([
                TransformerBlock(dim, num_heads, mlp_ratio,
                                 drop=drop_rate, attn_drop=0.0, drop_path=dpr[i])
                for i in range(depth)
            ])
            self.norm = nn.LayerNorm(dim)

        if use_metadata:
            assert use_vit, "metadata fusion needs the ViT trunk for the CLS query"
            self.meta_embed = MetadataEmbedding(dim)
            self.cross_attn = CrossAttentionBlock(dim, num_heads, mlp_ratio,
                                                  drop=drop_rate, attn_drop=0.0)

        # ── Classifier head ─────────────────────────────────────────────────
        # When use_vit=False we classify directly off the GAP'd CNN feature.
        head_in = dim if use_vit else feat_ch
        self.head_drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.head      = nn.Linear(head_in, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        self._init_transformer_weights()

    # -- weight init --------------------------------------------------------
    def _init_transformer_weights(self):
        """ViT-style truncated-normal init for all from-scratch Linear/Conv layers.

        The CNN stem keeps its pretrained ImageNet weights (we explicitly do
        NOT call .apply on it).
        """
        for name, m in self.named_modules():
            if name.startswith("stem"):       # leave pretrained weights alone
                continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # -- API expected by training/eval scripts ------------------------------
    def get_classifier(self) -> nn.Module:
        """Final linear classifier (kept for timm-compatible API)."""
        return self.head

    def get_param_groups(self, lr: float, backbone_lr_scale: float):
        """Two-group AdamW config for differential LR.

        - The pretrained CNN stem trains at `lr * backbone_lr_scale`
          (typically 0.3× — gentle fine-tuning).
        - Everything else (ViT trunk, metadata head, classifier) trains at
          full `lr` since it is initialised from scratch.

        train.py prefers this method when present; baselines that lack it
        fall back to the get_classifier()-based split.
        """
        stem_params = list(self.stem.parameters())
        stem_ids    = {id(p) for p in stem_params}
        scratch_params = [p for p in self.parameters() if id(p) not in stem_ids]
        return [
            {"params": stem_params,    "lr": lr * backbone_lr_scale, "name": "stem"},
            {"params": scratch_params, "lr": lr,                     "name": "scratch"},
        ]

    # -- forward ------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run image through CNN stem (+ ViT if enabled).

        Returns either:
          - (B, dim) when `use_vit=False` (GAP'd CNN feature), or
          - (B, dim) the CLS token after the ViT trunk.
        """
        feat = self.stem(x)                                                # (B, C, 14, 14)

        if not self.use_vit:
            return F.adaptive_avg_pool2d(feat, 1).flatten(1)               # (B, C)

        # Project + flatten to a token sequence (stride-2 conv: 14×14 → 7×7)
        feat = self.proj(feat)                                             # (B, dim, 7, 7)
        B = feat.shape[0]
        tokens = feat.flatten(2).transpose(1, 2)                           # (B, 49, dim)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed          # (B, 50, dim)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        return tokens[:, 0]                                                # CLS, (B, dim)

    def forward(self, x: torch.Tensor,
                meta: Optional[dict] = None) -> torch.Tensor:
        cls = self.forward_features(x)                                     # (B, dim) or (B, C)

        if self.use_vit and self.use_metadata:
            if meta is None:
                raise RuntimeError(
                    "HybridCNNViT(use_metadata=True) called without a `meta` dict."
                )
            cls = cls.unsqueeze(1)                                         # (B, 1, dim)
            meta_tokens = self.meta_embed(meta)                            # (B, 3, dim)
            cls = self.cross_attn(cls, meta_tokens).squeeze(1)             # (B, dim)

        return self.head(self.head_drop(cls))                              # (B, num_classes)


# ── Pure ViT (for the `hybrid_vit_only` ablation) ─────────────────────────────

class PureViT(nn.Module):
    """Standard ViT-Tiny variant operating directly on 16×16 image patches.

    No CNN stem — used only for the `hybrid_vit_only` ablation to isolate
    the benefit of the CNN-stem inductive bias. Same depth/width as
    `HybridCNNViT` for a controlled comparison.
    """

    def __init__(
        self,
        num_classes:    int = 8,
        img_size:       int = 224,
        patch_size:     int = 16,
        dim:            int = 192,
        depth:          int = 6,
        num_heads:      int = 4,
        mlp_ratio:    float = 2.0,
        drop_rate:    float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0.0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio,
                             drop=drop_rate, attn_drop=0.0, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head_drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.head      = nn.Linear(dim, num_classes)

        # ViT-style init
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_classifier(self) -> nn.Module:
        return self.head

    def get_param_groups(self, lr: float, backbone_lr_scale: float):
        # Everything is from scratch — single LR.
        return [{"params": list(self.parameters()), "lr": lr, "name": "all"}]

    def forward(self, x: torch.Tensor,
                meta: Optional[dict] = None) -> torch.Tensor:  # `meta` ignored
        del meta
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)             # (B, N, dim)
        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return self.head(self.head_drop(tokens[:, 0]))


# ── Factory + helpers ─────────────────────────────────────────────────────────

HYBRID_VARIANTS = {
    "hybrid_full",       # CNN + ViT + metadata fusion  (the proposed model)
    "hybrid_no_meta",    # CNN + ViT, no metadata        (ablation: meta)
    "hybrid_cnn_only",   # CNN stem + GAP + head         (ablation: ViT)
    "hybrid_vit_only",   # Pure ViT-Tiny, no CNN stem    (ablation: CNN)
}


def create_hybrid(name: str, pretrained: bool = True, num_classes: int = 8,
                  **kwargs) -> nn.Module:
    """Factory for the four hybrid-model variants. See `HYBRID_VARIANTS`."""
    if name == "hybrid_full":
        return HybridCNNViT(num_classes=num_classes, use_vit=True,
                            use_metadata=True, cnn_pretrained=pretrained, **kwargs)
    if name == "hybrid_no_meta":
        return HybridCNNViT(num_classes=num_classes, use_vit=True,
                            use_metadata=False, cnn_pretrained=pretrained, **kwargs)
    if name == "hybrid_cnn_only":
        return HybridCNNViT(num_classes=num_classes, use_vit=False,
                            use_metadata=False, cnn_pretrained=pretrained, **kwargs)
    if name == "hybrid_vit_only":
        return PureViT(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown hybrid variant {name!r}; must be one of {HYBRID_VARIANTS}")


def needs_metadata(name: str) -> bool:
    """True if `name` is a model that consumes the metadata dict."""
    return name == "hybrid_full"


def is_hybrid(name: str) -> bool:
    """True if `name` refers to any of the in-house hybrid/ViT variants."""
    return name in HYBRID_VARIANTS