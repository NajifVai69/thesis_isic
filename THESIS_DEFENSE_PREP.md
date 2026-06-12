# Thesis Defense Prep — Hybrid CNN–Transformer Networks with Clinical Metadata for Skin Lesion Classification

---

## 1. The 60-second pitch

"We built two models for ISIC-2019 skin lesion classification (8 classes, severely
imbalanced — NV is 53x larger than DF). Both share the same philosophy: CNN
features for local texture, a transformer trunk for global shape, and a
cross-attention head that injects clinical metadata (age, sex, lesion site) with
**no imputation** — missing fields get a learned 'missing' token.

- **Lightweight hybrid**: truncated MobileNetV2 + 6-layer ViT, **3.98M params,
  0.631 GMACs → 0.6081 BMA**. Best in its budget class, beats MobileViT-S and
  EfficientFormer-L1.
- **DEKAN (flagship)**: dual backbone (DenseNet-121 + EfficientNet-B0) fused via
  learned attention, 8-layer TinyViT trunk, KAN classifier head. **16.45M params,
  6.633 GMACs → 0.6438 BMA**, comparable to the top ISIC-2019 leaderboard
  *ensemble* (0.636) in a single model.

Primary metric is **BMA (balanced multi-class accuracy = macro-recall)** because
overall accuracy hides failure on rare malignant classes (DF, VASC, SCC, AK)."

---

## 2. Numbers you must have cold

| Item | Value |
|---|---|
| Dataset size (raw / after dedup) | 25,331 → 24,068 (removed 1,263 near-dups via pHash) |
| Split | 70/10/20 → 16,872 train / 2,400 val / 4,796 test |
| Classes | MEL, NV, BCC, AK, BKL, DF, VASC, SCC |
| Imbalance ratio | NV 12,875 vs DF 239 → **53.9x** |
| Metadata missingness | age ~30%, sex ~20%, site ~15% |
| Lightweight hybrid | 3.98M params, 0.631 GMACs, **BMA 0.6081**, macro-F1 0.5602, macro-AUC 0.9229, acc 0.7162 |
| DEKAN | 16.45M params, 6.633 GMACs, **BMA 0.6438**, macro-F1 0.6059, macro-AUC 0.9271, acc 0.7467 |
| dekan_linear (ablation) | **BMA 0.6510** — highest in study, but linear head not the flagship |
| CB-Focal Loss params | β=0.999, γ=2.5, label smoothing 0.1 |
| Optimizer | AdamW, lr 3e-4, cosine + 5-epoch warmup, grad clip 5 |
| EMA decay | 0.9998 |
| TTA | 8-view (hflip+vflip+4 rotations), softmax-averaged |
| Seeds | **single seed (42)** — explicitly flagged as a limitation |
| Training time | hybrid ≈3h, DEKAN ≈9h on RTX 4070 Ti Super |
| Leaderboard reference | DAISYLab ensemble ≈0.636 BMA (top ISIC-2019 entry) |

---

## 3. Architecture — be ready to draw/explain both from memory

### Lightweight Hybrid (Fig 4.1a)
1. **MobileNetV2 stem** (ImageNet pretrained, truncated at stride-16) → 96×14×14 feature map (~1.81M params)
2. **2×2 stride-2 conv** projects to 49 tokens (7×7), dim=192
3. Prepend CLS token + learned positional embedding → 50 tokens
4. **6 pre-norm transformer blocks**, 4 heads, MLP ratio 2, stochastic depth up to 0.1
5. **Metadata cross-attention**: image CLS token = query; 3 metadata tokens (age, sex, site) = keys/values. Age = linear projection or learned "missing" vector. Sex/site = learned categorical embeddings + "missing" vector. Each token also gets a learned type embedding.
6. **Linear classifier** → 8 logits

### DEKAN (Fig 4.1b)
1. Two parallel pretrained stems: **DenseNet-121** + **EfficientNet-B0**, both stride-16
2. Each backbone's feature map → 49 tokens, dim=256, tagged with per-backbone embedding
3. **Learned attention fusion**: a learnable bank of 49 query tokens cross-attends over the concatenation of both token sets (98 tokens) → fused 49-token sequence (this is the "DEKAN" fusion — content-dependent, not concat/average)
4. CLS + positional embedding → **8-layer TinyViT-style trunk** (dim 256, 8 heads, MLP ratio 4)
5. Same metadata cross-attention head
6. **KAN classifier** (Kolmogorov–Arnold layer, grid size 5, cubic B-splines, full precision)

### Why this design (be ready to justify)
- CNN-only and ViT-only ablations score nearly identically (~0.485–0.491 BMA) — but combined they jump to 0.608. **This is your strongest result** — the components are individually weak but strongly complementary.
- Pretrained CNN stems are needed because the transformer trains from scratch on only ~17k images.
- Stochastic depth/Mixup/RandAugment/EMA all exist specifically to stabilize the from-scratch transformer.

---

## 4. Anticipated questions & how to answer

### On the core claims
**Q: Why BMA instead of accuracy?**
A: With a 53x imbalance, a model predicting only NV gets ~50% accuracy while having 0% recall on DF/VASC/SCC — exactly the malignant/clinically critical classes. BMA = mean of per-class recalls, so every class counts equally regardless of frequency. This is also framed as an *ethical* requirement (NFR8) — rare malignant classes must not be invisible to the metric.

**Q: Your lightweight model (0.6081) is close to MobileViT-S (0.6020) — is the gain meaningful?**
A: It's a +0.006 gain at fewer params (3.98M vs 5.60M) and ~63% of the compute (0.631 vs 1.000 GMACs), so on a per-parameter and per-FLOP basis it's a clear win (BMA/M-param 0.153 vs 0.108). Also it's the *only* model in the sub-6M budget that beats both lightweight-hybrid baselines (MobileViT-S, EfficientFormer-L1) simultaneously. Frame it as "best Pareto point under budget," not "huge absolute jump."

**Q: DEKAN at 0.6438 vs the leaderboard's 0.636 — is that a fair comparison?**
A: No, and the thesis says so explicitly (Section 5.7) — the leaderboard number is on ISIC's official challenge server, ours is an internal held-out split. We present it as *context* ("our split is in the same ballpark as ensemble SOTA"), not a ranking claim.

### On the negative/inconclusive results (examiners will hunt these — own them)
**Q: Your best single number (0.6510, dekan_linear) isn't your "flagship" model. Doesn't that undermine the KAN contribution?**
A: Yes — and we report it transparently rather than hide it. The KAN head was a hypothesis (spline-based units might be more expressive at the classifier stage); the controlled ablation shows it does **not** help here (-0.007 vs linear). Our interpretation: by the time you reach the CLS token from a rich dual-backbone representation, the discriminative boundary is close to linear-separable, so KAN's extra expressivity doesn't pay for its added params/compute. This is reported as an honest negative result (a stated ethical/transparency requirement, NFR8/Section 3.4).

**Q: Does metadata actually help?**
A: Mixed signal, reported honestly:
- For the **lightweight hybrid**: validation BMA favored metadata by +0.011, but on the single test run the full model (0.6081) and no-metadata variant (0.6085) are within 0.0004 — inconclusive on a single evaluation.
- For **DEKAN**: dekan_linear (with metadata) = 0.6510 vs dekan_no_meta = 0.6302 → a clear **+0.021 BMA gain**. So metadata helps more clearly at higher capacity, suggesting the effect needs a model with enough capacity to exploit it / needs more seeds to detect at the lightweight scale.
- Bonus interesting finding: TTA × metadata interaction — without metadata, TTA *hurts* the hybrid (0.6216→0.6085), with metadata TTA *helps* (0.5815→0.6081). They converge to nearly the same point from opposite sides. Interpretation: metadata cross-attention regularizes per-view predictions, making them less individually overconfident, so averaging helps rather than hurts.

**Q: Single seed — how do you know these numbers aren't noise?**
A: We acknowledge this as an open limitation (risk register, Table 3.6) — "single-seed variance" is explicitly carried forward as future work requiring 3-seed mean±std (a hard constraint stated in CLAUDE.md / project plan too). What we *can* say: training curves (Figs 5.2, 5.3) show smooth convergence with no instability/collapse, which is at least evidence the result isn't a lucky/unlucky spike from an unstable run. But yes — formally, claims of "beats baseline X" should be read as point estimates pending multi-seed confirmation.

**Q: You didn't train the single-backbone DEKAN ablations (DenseNet-only, EffNet-only). Why does that matter?**
A: It means we can't yet decompose *how much* of DEKAN's gain comes from dual-backbone fusion vs. just having a bigger/different single backbone than the lightweight model. It's flagged as future work. If pressed: "the fusion mechanism (learned query-bank cross-attention over concatenated tokens) is the architectural novelty we'd want to isolate, and that requires those two missing runs."

**Q: No external validation (ISIC-2020) — doesn't that limit your generalization claim?**
A: Correct, explicitly listed as a limitation and as "open (future work)" in the risk register. The originally planned external eval (8-class softmax → mel-vs-rest on ISIC-2020, zero-shot binary) was not completed. Be upfront: all numbers are in-distribution on ISIC-2019's own split.

**Q: Skin tone / demographic bias — Bangladesh deployment context?**
A: ISIC-2019 over-represents fair (European) skin tones. We flag this as a genuine equity concern (Section 3.2/3.4), not solved — any real deployment in South Asia would need re-validation on a locally representative dataset. This is intentionally framed as a requirement for future work, not swept under the rug.

### On methodology / design choices
**Q: Why CB-Focal Loss with β=0.999 specifically (not the more aggressive β from the original paper)?**
A: We tried more aggressive β and it produced an extreme weight ratio between common and rare classes, causing rare classes to **collapse early in training**. β=0.999 gave stable optimization while still meaningfully reweighting. This was an empirical finding from preliminary runs (Table 5.2).

**Q: Why memory-mapped dataset / num_workers=0?**
A: Windows DataLoader with num_workers>0 deadlocks (well-known issue). Memmap means after epoch 1 the whole 3.6GB array sits in OS page cache, so a "load" is just a memcpy — recovers most of the throughput you'd lose from single-worker loading. This was a hard infrastructure constraint (16GB GPU, single Windows workstation) turned into a design decision.

**Q: Why pHash dedup + lesion-grouped split? What breaks without it?**
A: ISIC-2019 is a union of HAM10000/BCN20000/MSK with many near-duplicate images of the *same physical lesion* (different angles/zoom). Naive random splitting would put near-identical images of the same lesion in both train and test → leakage → inflated test scores that don't reflect generalization. pHash (Hamming distance ≤4, union-find) removed 1,263 near-dupes; the remaining split is grouped by lesion_id so no lesion straddles splits, and stratified by class×source so each split has a representative mix.

**Q: Why Shades-of-Gray color constancy specifically?**
A: Dermoscopic images come from different dermatoscope hardware/lighting across the three source archives (HAM/BCN/MSK), causing strong illumination/color shifts. Shades-of-Gray (Minkowski p=6) normalizes this once during preprocessing so the model doesn't have to learn to be invariant to scanner-specific color casts — reduces a confound, especially important since source correlates with class distribution.

**Q: Mixup + CB-Focal Loss together — isn't Mixup's soft-label mixing in tension with class-balanced reweighting?**
A: Mixup is applied to only half the batches, and **metrics are always computed on the original unmixed labels** — Mixup acts purely as a regularizer for the from-scratch transformer (preventing overfitting on ~17k images), it doesn't interact with the loss's class weighting at evaluation time. (Note: CutMix/MixUp were called out in CLAUDE.md as "hurts fine-grained dermoscopy" for a different reason — be ready if asked why this contradicts that note. Possible answer: this refers to the *low-strength, half-batch* Mixup used here for regularization vs. the heavier full-batch CutMix/MixUp combos that can blend lesion boundaries and destroy fine texture cues — the thesis used a tuned-down version specifically because of that risk.)

**Q: ResNet-18 dropped massively with TTA (-0.151). Why include such a bad result?**
A: For an honest, fair, single-protocol comparison — every model gets the same TTA treatment regardless of outcome. ResNet-18's failure illustrates a known TTA failure mode: if per-augmented-view predictions are confidently wrong in *different* directions (model isn't rotation/flip-robust), averaging softmax can make things worse rather than better. It's a useful negative data point that contextualizes why the proposed models' positive TTA deltas are notable (their architectures/training are more view-consistent).

### On societal/ethical/economic framing (Ch.3 — likely 1-2 questions from non-technical committee members)
**Q: Is this meant to replace a dermatologist?**
A: No — explicitly framed as decision support: ranked list of differential diagnoses (FR4), not a single hard label, to mitigate automation bias. Clinician stays in the loop.

**Q: What's the actual deployment story?**
A: Lightweight hybrid (3.98M params, 0.631 GMACs) can run on an edge device like a Jetson Orin Nano (~$249, 7-25W) — no internet needed, relevant for low-connectivity rural clinics. 3-yr TCO ~$258 vs ~$4,380 for cloud T4 inference. DEKAN is the higher-accuracy option for settings where compute budget is less constrained.

**Q: Carbon footprint?**
A: Training both models ≈3 + 2.25 = ~3 kWh combined for the headline runs (~0.47 + 1.40 kg CO2eq); whole-project total across baselines/ablations is "tens of kWh" / "tens of kg CO2eq" — roughly one short car trip. Orders of magnitude below training the heavy ensembles that top the leaderboard.

---

## 5. Per-class story (good for "what does the model actually get wrong" questions)

- Lightweight hybrid's **best class vs all models studied**: AK recall = 0.587 (highest, even above DEKAN's 0.368). Hypothesis: AK is age/sun-exposure correlated and metadata cross-attention shifts the prior for older patients / sun-exposed sites.
- Lightweight hybrid's **worst class**: SCC recall = 0.223 — confused with AK/BKL (all keratinocytic lesions, visually similar).
- DEKAN **recovers SCC substantially** (0.529) and gets the best VASC (0.922) and BCC (0.731) recalls — these drive its higher overall BMA, but it *loses* on AK (0.368 vs hybrid's 0.587).
- Confusion matrices (Figs 5.5, 5.6): most off-diagonal mass is within the **keratinocytic group (AK/BKL/SCC)** for both models — this is the genuinely hard sub-problem, consistent with real dermatological difficulty (these lesions are visually similar even to specialists).

---

## 6. Likely "gotcha" / synthesis questions

**Q: If you had to pick ONE model to deploy in a rural Bangladesh clinic, which and why?**
A: Lightweight hybrid — edge-deployable, offline, $249 hardware, 0.6081 BMA is "good enough" for triage/referral (decision support, not diagnosis), and the cost gap to DEKAN (16.45M params, 10x compute) isn't justified when the system's role is flagging-for-review rather than final diagnosis. DEKAN would suit a referral hospital with more compute budget where the +0.036 BMA (or +0.043 with dekan_linear) matters more.

**Q: What's the single most important thing you'd do with one more month?**
A: Multi-seed runs (≥3 seeds, mean±std) for the two proposed models — it's the cheapest way to convert "0.6081 vs 0.6020" from a point estimate into a defensible statistical claim, and it's the limitation most likely to be challenged by reviewers/examiners.

**Q: What would you do differently if starting over?**
A: Possible honest answers: (1) budget seed-repeats into the schedule from the start rather than as "if time permits"; (2) train the missing single-backbone DEKAN ablations alongside the full model since they share most of the pipeline; (3) consider a smaller/faster external validation set early so it's not dropped under time pressure.

**Q: Novelty check — what's actually new here vs. Pacheco & Krohling?**
A: They fused metadata onto *heavy* CNNs via attention. We (1) apply the same "no imputation, learned missing-token" principle to a *lightweight* hybrid under an explicit <6M param/<1 GFLOP budget, and (2) extend it to a *dual-backbone* hybrid (DEKAN) with a novel learned-query-bank fusion mechanism + KAN head — and report an explicit efficiency claim (params/GMACs/BMA-per-param) that prior metadata-fusion work didn't focus on.

---

## 7. Quick defense logistics checklist

- [ ] Confirm exact macro-F1/macro-AUC numbers from `figures/summary_table.csv` match the thesis tables (you have this file open — worth a final cross-check before defense).
- [ ] Have Figures 5.1 (model comparison), 5.5/5.6 (confusion matrices), 5.8/5.9 (ablations), 5.7 (efficiency scatter) ready to pull up quickly if asked to "show me."
- [ ] Be ready to whiteboard Fig 4.1 (both architectures) from memory.
- [ ] Memorize the **two honest negative results** verbatim — examiners often probe exactly the things a thesis admits it didn't nail, to test whether the candidate understands *why*, not just *what*.
- [ ] Know which of the 5 team members can speak to which chapter (preprocessing vs. architecture vs. training vs. analysis vs. writing) in case the panel splits questions across the group.
