"""DEKAN — Dual-backbone CNN × TinyViT-style trunk × KAN classifier.

Architecture (dekan_full, the proposed model):

    Image (3 × 224 × 224)
        ├─ DenseNet-121 stem  (timm, features_only, ImageNet-pretrained, stride-16)
        │     → (B, C_d, 14, 14)   [C_d ≈ 256]
        └─ EfficientNet-B0 stem (timm, features_only, ImageNet-pretrained, stride-16)
              → (B, C_e, 14, 14)   [C_e ≈ 80]
            │
            │  Per-backbone stride-2 patch projection: 14×14 → 7×7
            │  + per-backbone type embedding + shared 2D pos embedding
            │  → T_d, T_e  each (B, 49, D)
            │
        Attention fusion
            │  Learnable query bank Q (B, 49, D)
            │  cross-attends over concat(T_d, T_e) (B, 98, D)  [CrossAttentionBlock]
            │  → fused sequence F (B, 49, D)
            │
        Prepend CLS + pos embed → (B, 50, D)
            │
        TinyViT-style trunk: depth × TransformerBlock(D, heads, mlp_ratio=4)
            │
        LayerNorm → CLS token → (B, D)
            │
        Metadata cross-attention (optional)
            │  CLS (B, 1, D) attends MetadataEmbedding({age, sex, site}) (B, 3, D)
            │  → (B, D)
            │
        KAN head  [or nn.Linear for ablation]  → (B, 8)

Budget (dekan_full at default D=256, depth=8) — verified from training output:
    NOTE: timm features_only loads FULL backbone weights for both stems
    (including blocks not reached in the forward pass), same as hybrid_full.
    DenseNet-121 stem (timm features_only) : ~6.95 M params,  ~0.44 GMAC
    EfficientNet-B0 stem (timm features_only):~1.73 M params,  ~0.15 GMAC
    Patch projections + type/pos embeds    :  ~0.34 M params
    Attention fusion block                 :  ~0.53 M params
    Transformer trunk (8 × D=256 mlp×4)   :  ~6.32 M params,  ~0.13 GMAC
    Metadata cross-attention + MetaEmbed   :  ~0.53 M params
    KAN head (KANLinear 256→8)             :  ~18  K params
    ──────────────────────────────────────────────────────────────
    Total (exact, from training script)    : 16.45 M params,  ~0.7 GMAC

Ablation variants (DEKAN_VARIANTS):
    dekan_full           dual-CNN + fusion + metadata + KAN   (proposed)
    dekan_no_meta        dual-CNN + fusion, no metadata       (ablation: metadata)
    dekan_linear         dekan_full but nn.Linear head        (ablation: KAN)
    dekan_densenet_only  DenseNet only + metadata + KAN       (ablation: fusion / EffNet)
    dekan_effnet_only    EffNet-B0 only + metadata + KAN      (ablation: fusion / DenseNet)

Reuses from models/hybrid.py:
    TransformerBlock, CrossAttentionBlock, MultiheadAttention, MetadataEmbedding

References:
    Huang et al.    "Densely Connected Convolutional Networks"   CVPR 2017
    Tan & Le        "EfficientNet: Rethinking Model Scaling …"   ICML 2019
    Liu et al.      "KAN: Kolmogorov-Arnold Networks"            arXiv 2404.19756
    Dosovitskiy +al."An Image is Worth 16×16 Words"              ICLR 2021
    Pacheco & Krohling "An attention-based mechanism …"          Comp.Biol.Med 2021
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hybrid import (
    CrossAttentionBlock,
    MetadataEmbedding,
    TransformerBlock,
)
from models.kan import KANLinear


# ── CNN stems ─────────────────────────────────────────────────────────────────

class _SingleCNNStem(nn.Module):
    """One timm backbone in features-only mode, producing the stride-16 feature map.

    Automatically identifies the output stage whose spatial reduction factor is 16
    (i.e. a 224×224 input becomes 14×14). Falls back to the largest reduction ≤ 16
    if no exact stride-16 stage exists in the given backbone.
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        import timm

        # Probe (no-weight) to discover which feature stage is stride-16.
        probe = timm.create_model(model_name, pretrained=False, features_only=True)
        all_reductions = [fi["reduction"] for fi in probe.feature_info.info]
        del probe

        if 16 in all_reductions:
            idx = all_reductions.index(16)
        else:
            valid = [i for i, r in enumerate(all_reductions) if r <= 16]
            idx   = valid[-1] if valid else 0

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(idx,),
        )
        self.out_channels = self.backbone.feature_info.channels()[-1]
        self.reduction    = self.backbone.feature_info.reduction()[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)[-1]    # (B, out_channels, H/16, W/16)


# ── DEKAN model ───────────────────────────────────────────────────────────────

class DEKAN(nn.Module):
    """Dual-backbone CNN + TinyViT-style trunk + KAN classifier.

    Args:
        num_classes:    Number of output classes (8 for ISIC-2019).
        dim:            Token / embedding dimension D.
        depth:          Number of TinyViT-style transformer blocks.
        num_heads:      Attention heads per block (must divide dim).
        mlp_ratio:      MLP hidden-dim multiplier in transformer blocks.
        drop_rate:      Dropout in MLP / projection layers.
        drop_path_rate: Peak stochastic-depth rate (linearly scaled per block).
        use_metadata:   Feed clinical metadata through a cross-attention head.
        use_kan:        Use a KANLinear head. If False, use nn.Linear (ablation).
        use_densenet:   Include the DenseNet-121 backbone branch.
        use_effnet:     Include the EfficientNet-B0 backbone branch.
        cnn_pretrained: Load ImageNet-1k weights for the CNN backbone(s).
    """

    def __init__(
        self,
        num_classes:    int   = 8,
        dim:            int   = 256,
        depth:          int   = 8,
        num_heads:      int   = 8,
        mlp_ratio:      float = 4.0,
        drop_rate:      float = 0.0,
        drop_path_rate: float = 0.1,
        use_metadata:   bool  = True,
        use_kan:        bool  = True,
        use_densenet:   bool  = True,
        use_effnet:     bool  = True,
        cnn_pretrained: bool  = True,
    ):
        super().__init__()
        assert use_densenet or use_effnet, "at least one CNN backbone must be enabled"
        self.use_fusion   = use_densenet and use_effnet   # fusion only when both present
        self.use_metadata = use_metadata
        self.use_kan      = use_kan
        self.use_densenet = use_densenet
        self.use_effnet   = use_effnet
        self.dim          = dim
        self.num_patches  = 7 * 7   # 14×14 feature map → stride-2 proj → 7×7 = 49 tokens

        # ── CNN stems + patch projections ──────────────────────────────────
        if use_densenet:
            self.densenet_stem = _SingleCNNStem("densenet121", pretrained=cnn_pretrained)
            # kernel_size=2, stride=2: folds 14×14 into 7×7 patches, doubles receptive field
            self.proj_d = nn.Conv2d(
                self.densenet_stem.out_channels, dim, kernel_size=2, stride=2, bias=True
            )

        if use_effnet:
            self.effnet_stem = _SingleCNNStem("efficientnet_b0", pretrained=cnn_pretrained)
            self.proj_e = nn.Conv2d(
                self.effnet_stem.out_channels, dim, kernel_size=2, stride=2, bias=True
            )

        # ── Backbone-type embeddings ────────────────────────────────────────
        # One (1, 1, D) learned vector per active backbone; added to every token
        # from that backbone so the transformer / fusion block can distinguish
        # which backbone produced each token.
        n_backbones = int(use_densenet) + int(use_effnet)
        self.backbone_type_embed = nn.Parameter(torch.zeros(n_backbones, 1, 1, dim))
        nn.init.trunc_normal_(self.backbone_type_embed, std=0.02)

        # ── Attention fusion block (dual-backbone path only) ────────────────
        if self.use_fusion:
            # Learnable query bank: one query token per spatial position (49 total).
            # These are updated end-to-end and represent "what to ask" the two
            # backbone token pools.  Inspired by slot / perceiver attention.
            self.fusion_query = nn.Parameter(torch.zeros(1, self.num_patches, dim))
            nn.init.trunc_normal_(self.fusion_query, std=0.02)
            # Cross-attend: Q = fusion_query, KV = concat(T_d, T_e) [98 tokens]
            self.fusion_block = CrossAttentionBlock(
                dim, num_heads, mlp_ratio=2.0, drop=drop_rate
            )

        # ── CLS token + positional embedding ───────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── TinyViT-style transformer trunk ─────────────────────────────────
        # Linear drop-path schedule: 0 at block 0 → drop_path_rate at last block.
        dpr = [x.item() for x in torch.linspace(0.0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim, num_heads, mlp_ratio,
                drop=drop_rate, attn_drop=0.0, drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # ── Clinical metadata cross-attention head ──────────────────────────
        if use_metadata:
            self.meta_embed = MetadataEmbedding(dim)
            self.cross_attn = CrossAttentionBlock(
                dim, num_heads, mlp_ratio=2.0, drop=drop_rate
            )

        # ── Classifier head ──────────────────────────────────────────────────
        self.head_drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()
        if use_kan:
            self.head = KANLinear(dim, num_classes, grid_size=5, spline_order=3)
        else:
            self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    # ── Weight init ─────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """ViT-style truncated-normal init for all from-scratch layers.

        Explicitly skips the two pretrained CNN stems so their ImageNet weights
        are preserved at the start of training.
        """
        pretrained_prefixes = {"densenet_stem", "effnet_stem"}
        for name, m in self.named_modules():
            if any(name.startswith(p) for p in pretrained_prefixes):
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

    # ── API expected by training/eval scripts ────────────────────────────────

    def get_classifier(self) -> nn.Module:
        """Final classifier (kept for timm-compatible API)."""
        return self.head

    def get_param_groups(self, lr: float, backbone_lr_scale: float) -> list[dict]:
        """Two-group AdamW config for differential learning rates.

        Pretrained CNN stems are gentle-finetuned at `lr * backbone_lr_scale`
        (default 0.3×).  All from-scratch components (fusion, trunk, head, …)
        train at full `lr`.
        """
        stem_ids: set[int] = set()
        if self.use_densenet:
            stem_ids.update(id(p) for p in self.densenet_stem.parameters())
        if self.use_effnet:
            stem_ids.update(id(p) for p in self.effnet_stem.parameters())

        stem_params    = [p for p in self.parameters() if id(p) in stem_ids]
        scratch_params = [p for p in self.parameters() if id(p) not in stem_ids]
        return [
            {"params": stem_params,    "lr": lr * backbone_lr_scale, "name": "stems"},
            {"params": scratch_params, "lr": lr,                     "name": "scratch"},
        ]

    # ── Forward ──────────────────────────────────────────────────────────────

    def _encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Run CNN stems, project to patches, and apply attention fusion.

        Returns (B, num_patches, dim) fused token sequence.
        """
        B = x.shape[0]
        tokens_list: list[torch.Tensor] = []
        bi = 0   # index into backbone_type_embed

        if self.use_densenet:
            feat_d = self.densenet_stem(x)                  # (B, C_d, 14, 14)
            tok_d  = self.proj_d(feat_d)                    # (B, D, 7, 7)
            tok_d  = tok_d.flatten(2).transpose(1, 2)       # (B, 49, D)
            tok_d  = tok_d + self.backbone_type_embed[bi]   # (1,1,D) broadcasts → (B,49,D)
            tokens_list.append(tok_d)
            bi += 1

        if self.use_effnet:
            feat_e = self.effnet_stem(x)                    # (B, C_e, 14, 14)
            tok_e  = self.proj_e(feat_e)                    # (B, D, 7, 7)
            tok_e  = tok_e.flatten(2).transpose(1, 2)       # (B, 49, D)
            tok_e  = tok_e + self.backbone_type_embed[bi]   # (1,1,D) → (B,49,D)
            tokens_list.append(tok_e)

        if self.use_fusion:
            kv     = torch.cat(tokens_list, dim=1)          # (B, 98, D)
            q      = self.fusion_query.expand(B, -1, -1)    # (B, 49, D)
            tokens = self.fusion_block(q, kv)               # (B, 49, D)
        else:
            tokens = tokens_list[0]                         # (B, 49, D) single backbone

        return tokens

    def forward(
        self,
        x:    torch.Tensor,
        meta: Optional[dict] = None,
    ) -> torch.Tensor:
        B = x.shape[0]

        # ── CNN stems → fused token sequence ───────────────────────────────
        tokens = self._encode_to_tokens(x)                  # (B, 49, D)

        # ── Prepend CLS + positional embedding ─────────────────────────────
        cls    = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)            # (B, 50, D)
        tokens = tokens + self.pos_embed                    # broadcast add

        # ── TinyViT-style transformer trunk ─────────────────────────────────
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]                              # CLS, (B, D)

        # ── Clinical metadata cross-attention ───────────────────────────────
        if self.use_metadata:
            if meta is None:
                raise RuntimeError(
                    "DEKAN(use_metadata=True) was called without a `meta` dict. "
                    "Pass use_metadata=False for the no-metadata ablation, or "
                    "ensure the DataLoader returns metadata."
                )
            cls_out      = cls_out.unsqueeze(1)             # (B, 1, D)
            meta_tokens  = self.meta_embed(meta)            # (B, 3, D)
            cls_out      = self.cross_attn(cls_out, meta_tokens).squeeze(1)  # (B, D)

        # ── Classifier head ──────────────────────────────────────────────────
        return self.head(self.head_drop(cls_out))           # (B, num_classes)


# ── Factory + helpers ─────────────────────────────────────────────────────────

DEKAN_VARIANTS: set[str] = {
    "dekan_full",           # dual-CNN + fusion + metadata + KAN   (proposed)
    "dekan_no_meta",        # dual-CNN + fusion, no metadata        (ablation: metadata)
    "dekan_linear",         # dekan_full but nn.Linear head         (ablation: KAN)
    "dekan_densenet_only",  # DenseNet only + metadata + KAN        (ablation: fusion / EffNet)
    "dekan_effnet_only",    # EfficientNet-B0 only + metadata + KAN (ablation: fusion / DenseNet)
}

# Shared hyperparameter defaults — all variants use the same trunk width/depth.
_DEKAN_DEFAULTS: dict = dict(
    dim=256, depth=8, num_heads=8, mlp_ratio=4.0,
    drop_rate=0.0, drop_path_rate=0.1,
)


def create_dekan(
    name: str,
    pretrained:  bool = True,
    num_classes: int  = 8,
    **kwargs,
) -> nn.Module:
    """Factory for DEKAN ablation variants.  See `DEKAN_VARIANTS` for names."""
    hp = {**_DEKAN_DEFAULTS, **kwargs}   # caller can override any hyperparameter

    if name == "dekan_full":
        return DEKAN(num_classes=num_classes,
                     use_metadata=True,  use_kan=True,
                     use_densenet=True,  use_effnet=True,
                     cnn_pretrained=pretrained, **hp)

    if name == "dekan_no_meta":
        return DEKAN(num_classes=num_classes,
                     use_metadata=False, use_kan=True,
                     use_densenet=True,  use_effnet=True,
                     cnn_pretrained=pretrained, **hp)

    if name == "dekan_linear":
        return DEKAN(num_classes=num_classes,
                     use_metadata=True,  use_kan=False,
                     use_densenet=True,  use_effnet=True,
                     cnn_pretrained=pretrained, **hp)

    if name == "dekan_densenet_only":
        return DEKAN(num_classes=num_classes,
                     use_metadata=True,  use_kan=True,
                     use_densenet=True,  use_effnet=False,
                     cnn_pretrained=pretrained, **hp)

    if name == "dekan_effnet_only":
        return DEKAN(num_classes=num_classes,
                     use_metadata=True,  use_kan=True,
                     use_densenet=False, use_effnet=True,
                     cnn_pretrained=pretrained, **hp)

    raise ValueError(
        f"Unknown DEKAN variant {name!r}; must be one of {DEKAN_VARIANTS}"
    )


def is_dekan(name: str) -> bool:
    """True if `name` refers to any DEKAN variant."""
    return name in DEKAN_VARIANTS


def dekan_needs_metadata(name: str) -> bool:
    """True if this DEKAN variant consumes the clinical metadata dict."""
    # dekan_no_meta is the only variant that disables the metadata head.
    return name in DEKAN_VARIANTS and name != "dekan_no_meta"
