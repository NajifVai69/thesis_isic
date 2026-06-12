# Thesis Defense — Presentation Plan (15 minutes, 5 speakers, ~3 min each)

20 slides total. Slide 13 is a blank section divider — skip it (don't speak).
Practice with a timer. If running long, cut the *italic optional* lines first.

---

## SAYEM'S SPEECH — Slides 1–4 (Simple Version)

This is an easier, more conversational version of slides 1–4 for Sayem to
deliver. Just talk naturally — these are short, simple sentences you can
say in your own words.

---

**[SLIDE 1 — Title Slide]**

> "Good morning everyone. Our thesis is called 'Hybrid CNN-Transformer
> Networks with Clinical Metadata for Skin Lesion Classification.' Our
> supervisor is Dr. Md. Ashraful Alam. I'm Sayem, and with me today are
> Nazif, Samin, Munia, and Shakib."

*(Just say this quickly — about 10 seconds — then move to the next slide.)*

---

**[SLIDE 2 — Table of Contents]**

> "Quickly, here's what we'll cover: why this problem matters, our dataset,
> our two models, our results, and finally our limitations and conclusions."

*(About 5–10 seconds. Don't read every bullet — just give the overview.)*

---

**[SLIDE 3 — Background and Motivation]**

> "So why does this matter? Early detection of skin cancer saves lives.
> Doctors use a special camera called a dermoscope to look closely at skin
> lesions, but reading these images correctly needs an expert — and not
> every clinic has one.
>
> Our idea is to combine three things in one AI model:
>
> First — CNNs, which are good at spotting small details like texture and
> patterns on the skin.
>
> Second — Transformers, which are good at looking at the bigger picture —
> the overall shape and border of the lesion.
>
> Third — clinical information that's already collected, like the patient's
> age, sex, and where on the body the lesion is. Doctors use this
> information naturally, but most AI models simply ignore it.
>
> So our goal was simple: build a model that's small enough to actually run
> in a real clinic, but accurate enough to correctly identify all eight types
> of skin lesions — not just the common ones."

*(About 45–50 seconds. Speak slowly and clearly — this slide sets up
everything else.)*

---

**[SLIDE 4 — Problem Statement & Research Gap]**

> "Now, the main challenge. Our dataset is very imbalanced. One type of
> lesion, called NV, has almost 13,000 images. But another type, called DF,
> has only 239 images. That's a huge difference — more than 50 times more
> images for one class than the other.
>
> Why does this matter? If we just look at overall accuracy, a lazy model
> could just guess the common class — NV — every single time, and still get
> a decent score. But it would completely fail on the rare classes — and
> those rare classes are often the ones that matter most clinically.
>
> Two more problems we noticed: most top-performing systems on this dataset
> use huge combinations of many models together — which isn't practical for
> real-world use. And most existing models only look at the image, throwing
> away useful patient information.
>
> Because of this, instead of using regular accuracy, we used a fairer
> measurement called Balanced Multi-class Accuracy — basically, the average
> accuracy across all eight classes treated equally, so rare classes count
> just as much as common ones."

*(About 45–50 seconds. Take your time on the imbalance numbers — 12,875 vs
239 — that's the key fact examiners will remember.)*

---

**Total time for slides 1–4: roughly 2 minutes.** That leaves you a little
buffer if you pause, get a question, or speak a bit slower.

**Quick tips:**
- You don't need to memorize word-for-word — just know the *flow*: intro →
  why this matters → the big problem (imbalance) → why we measure things
  differently (BMA).
- The two numbers to nail: **12,875 (NV) vs 239 (DF)** — that's your biggest
  "wow" fact, say it clearly and slowly.
- If someone asks "what is BMA again?" during Q&A: "It's the average of how
  well the model recognizes each of the 8 classes, so rare diseases count
  just as much as common ones."

---

## SPEAKER 1 — Nazif Bin Morshed
**Slides 1–4 | Target: ~3:00**

### Slide 1 — Title
> "Good [morning/afternoon] everyone. Our thesis is titled 'Hybrid CNN-Transformer
> Networks with Clinical Metadata for Skin Lesion Classification,' supervised by
> Dr. Md. Ashraful Alam. I'm [name], and with me are [names of other 4]."

*(~10 sec — keep it short, just introduce the team and move on)*

### Slide 2 — Table of Contents
> "Here's the outline of our talk: motivation and the problem we're solving,
> our dataset and preprocessing, our two proposed models, results, ablations,
> limitations, and conclusions."

*(~10 sec)*

### Slide 3 — Background and Motivation
> "Skin cancer outcomes depend heavily on early detection. Dermoscopy — imaging
> the skin with magnification and polarized light — reveals structures beneath
> the surface, but interpreting these images correctly requires specialist
> training that isn't always available.
>
> Our approach combines three complementary signals: CNNs are excellent at
> picking up local texture patterns — the fine pigment networks and dot
> patterns dermatologists look for. Transformers are better at modeling global
> structure — things like overall asymmetry and border irregularity, which are
> two of the ABCD criteria clinicians use. And third, clinical context — a
> patient's age, sex, and where on the body the lesion is — is information
> dermatologists routinely use but most AI systems completely ignore.
>
> Our goal was to combine all three into a model that's small enough to
> actually deploy, but accurate enough across all eight diagnostic classes —
> not just the common ones."

*(~50 sec)*

### Slide 4 — Problem Statement & Research Gap
> "The core challenge is severe class imbalance. In ISIC-2019, the most common
> class — melanocytic nevus, or NV — has nearly 13,000 images, while
> dermatofibroma, DF, has only 239. That's over a 50-times difference.
>
> This matters because a naive model can score high on overall accuracy just
> by always predicting the common classes, while completely failing on the
> rare ones — which are often the clinically critical, potentially malignant
> ones.
>
> Two more gaps: most high-performing ISIC systems rely on heavy ensembles of
> many large models — not practical for real deployment. And most image-only
> classifiers throw away metadata that's already collected and that
> dermatologists actually use.
>
> Because of this imbalance, we chose Balanced Multi-class Accuracy — the mean
> recall across all eight classes — as our primary metric, instead of overall
> accuracy."

*(~50 sec)*

**Total: ~2:00–2:10** (gives buffer for slide 1/2 intros running slightly long)

---

## SPEAKER 2 — Md. Samin Yaser
**Slides 5–7 | Target: ~3:00**

### Slide 5 — Objectives
> "Building on that motivation, we set four concrete objectives.
>
> First, design a lightweight CNN-Transformer classifier under a strict budget
> — fewer than 6 million parameters and under 1 GMAC of compute.
>
> Second, fuse age, sex, and anatomical site into the model using metadata
> cross-attention, with learned 'missing' tokens — so we never need to impute
> missing values.
>
> Third, develop a second, higher-capacity model — we call it DEKAN — a
> dual-backbone design aimed at pushing balanced accuracy further.
>
> And fourth, compare both models fairly against five established baselines,
> all trained under the exact same protocol, and quantify our design choices
> through ablation studies.
>
> To preview the headline numbers: our lightweight model hits 3.98 million
> parameters with a BMA of 0.6081, and DEKAN reaches 16.45 million parameters
> with a BMA of 0.6438."

*(~45 sec)*

### Slide 6 — Dataset: ISIC-2019
> "We used ISIC-2019 — 25,331 labeled dermoscopic images across 8 diagnostic
> classes, drawn from three sources: HAM10000, BCN20000, and MSK.
>
> Alongside each image, we have metadata: approximate age, sex, anatomical
> site, and a lesion ID. Importantly, this metadata has real-world
> missingness — about 30% of age values, 20% of sex values, and 15% of site
> values are missing. We treat this missingness as meaningful information
> rather than something to fill in.
>
> As you can see in the class distribution, NV alone has nearly 13,000 images
> while DF and VASC have only around 250 each — the largest class is more than
> 50 times the size of the smallest."

*(~35 sec)*

### Slide 7 — Dataset Pre-processing
> "Before training, we ran a four-stage preprocessing pipeline.
>
> First, perceptual hash, or pHash, near-duplicate removal — this caught 1,263
> near-identical images, mostly different photos of the exact same lesion,
> leaving 24,068 unique images.
>
> Second, a lesion-grouped, stratified 70/10/20 train/validation/test split.
> The key reason we group by lesion ID is to prevent train-test leakage — if
> two near-duplicate photos of the same lesion ended up in different splits,
> the model could effectively 'memorize' the test answer.
>
> Third, color constancy normalization to reduce lighting and device
> differences across the three source datasets, followed by resizing to
> 224×224.
>
> Finally, everything is packed into a memory-mapped array for fast training —
> this was also driven by our hardware constraint of a single 16GB GPU."

*(~40 sec)*

**Total: ~2:00**

---

## SPEAKER 3 — Sayem Afridi
**Slides 8–10 | Target: ~3:00**

### Slide 8 — Proposed Framework Overview
> "Both of our proposed models follow the same design philosophy at two
> different capacity points.
>
> A CNN stem extracts local texture features — the fine-grained patterns
> dermatologists look at up close. A transformer trunk then operates on those
> features to model global lesion structure — overall shape, asymmetry, and
> borders. And a metadata cross-attention module folds in age, sex, and
> anatomical site as additional context.
>
> The first tier is our compact, deployment-ready Lightweight Hybrid. The
> second is DEKAN, our accuracy-focused flagship."

*(~30 sec)*

### Slide 9 — Lightweight Hybrid CNN-Transformer
> "The Lightweight Hybrid starts with a 224×224 image, passed through a
> truncated, ImageNet-pretrained MobileNetV2 stem — we keep only the early
> layers that capture generic edge and texture features. This produces a
> feature map that we project into 49 tokens.
>
> Those tokens go through a 6-layer Vision Transformer with 4 attention
> heads, which models global structure across the lesion.
>
> Then comes our metadata cross-attention block: the image's CLS token attends
> over three metadata tokens — age, sex, and anatomical site. Whenever a value
> is missing, instead of imputing a guess, we use a learned 'missing'
> embedding — so the model explicitly knows 'this information isn't
> available' rather than being given a potentially misleading default.
>
> Finally, a simple linear classifier produces the 8-class prediction.
>
> The result: 3.98 million parameters, 0.631 GMACs — comfortably under our
> 6-million, 1-GMAC budget — and a BMA of 0.6081 with test-time augmentation.
> This is the best balanced accuracy of any model we tested under the
> 6-million-parameter budget."

*(~55 sec)*

### Slide 10 — DEKAN: Dual-Backbone Flagship Model
> "DEKAN is our higher-capacity model. It runs two pretrained CNN stems in
> parallel — DenseNet-121 and EfficientNet-B0 — to get two complementary
> feature representations of the same image.
>
> These are combined using a learned attention fusion module — rather than
> simply concatenating or averaging the two feature sets, the model learns how
> to weight and combine them.
>
> The fused tokens then go through a larger, 8-layer TinyViT-style transformer
> with 256-dimensional embeddings and 8 attention heads — followed by the same
> metadata cross-attention mechanism as the lightweight model.
>
> For the final classifier, we used a KAN — Kolmogorov-Arnold Network — layer,
> which uses learnable spline functions instead of fixed weight-and-activation
> combinations.
>
> DEKAN comes in at 16.45 million parameters and 6.633 GMACs, achieving a BMA
> of 0.6438 with TTA and a macro-AUC of 0.9271 — our best result among the main
> proposed models, at a higher computational cost."

*(~50 sec)*

**Total: ~2:15–2:20**

---

## SPEAKER 4 — Mos. Mahabuba Akter Munia
**Slides 11, 12, 14, 15 | Target: ~3:00** *(slide 13 is blank, skip it)*

### Slide 11 — Training and Evaluation Protocol
> "To make our comparison fair, every model — baselines and proposed —
> was trained with the exact same recipe.
>
> We used AdamW with a base learning rate of 3e-4, gradient clipping at 5, a
> 5-epoch warmup followed by cosine annealing down to 1e-6.
>
> For the loss function, we used Class-Balanced Focal Loss with beta=0.999 and
> gamma=2.5, plus label smoothing of 0.1 — this directly addresses the class
> imbalance we discussed earlier.
>
> Augmentation included flips, random resized crops, color jitter, RandAugment
> and Mixup. For efficiency, we used automatic mixed precision, channels-last
> memory format, and EMA — exponential moving average — weights for
> validation.
>
> Our primary metric is BMA — mean per-class recall — with macro-F1,
> macro-AUC, and accuracy as secondary metrics. At inference, we used 8-view
> test-time augmentation, averaging softmax outputs across flips and
> rotations."

*(~45 sec)*

### Slide 12 — Main Result: ISIC-2019 Test Set
> "Here are our headline results. ResNet-18 scored just 0.337 BMA — it
> actually got *worse* with TTA, which we'll come back to. MobileNetV2 reached
> 0.474, EfficientNet-B0 0.495. The lightweight-hybrid baselines did better:
> MobileViT-S at 0.602 and EfficientFormer-L1 at 0.552.
>
> Our Lightweight Hybrid reached 0.608 — the best BMA of any model under the
> 6-million-parameter budget, beating both lightweight-hybrid baselines.
>
> DEKAN reached 0.644 — the best BMA overall, leading on every metric we
> measured, but at higher parameter and compute cost.
>
> So both of our proposed models outperform the CNN-only baselines on balanced
> accuracy."

*(~35 sec)*

### Slide 14 — Per-Class Recall Analysis
> "Looking at performance class-by-class tells a more complete story. Our
> Hybrid model strengthens the harder 'middle' classes — BCC, AK, and BKL —
> compared to EfficientNet-B0 alone.
>
> DEKAN recovers SCC substantially, going from the Hybrid's 0.223 up to 0.529,
> and achieves the strongest VASC recall at 0.922 and BCC recall at 0.731.
>
> The remaining error cluster is concentrated among the keratinocytic classes
> — AK, BKL, and SCC. These lesion types are visually very similar even to
> trained dermatologists, so this is a genuinely hard sub-problem, not just a
> model weakness."

*(~30 sec)*

### Slide 15 — Efficiency vs Accuracy Trade-off
> "Putting accuracy and parameter count together: our Hybrid model is the most
> parameter-efficient model in the sub-6-million budget, with a BMA-per-million-
> parameters of 0.153 — more than 3 times better than DEKAN's 0.039.
>
> DEKAN deliberately trades that efficiency for the highest absolute BMA.
> Together, these give us two deployment options: a compact, clinic-friendly
> model, or an accuracy-focused model for settings with more compute."

*(~25 sec)*

**Total: ~2:15**

---

## SPEAKER 5 — Nazmus Shakib
**Slides 16–20 | Target: ~3:30** *(slightly longer — closing speaker, can absorb overflow)*

### Slide 16 — TTA and Ablation Insights
> "Our ablation studies tell us what actually mattered.
>
> For the Lightweight Hybrid, going from the CNN stem alone to the full hybrid
> improved BMA from 0.4853 to 0.6081 — a gain of +0.123. That's our strongest
> evidence that the CNN-Transformer combination is genuinely complementary,
> not just one component carrying the result.
>
> The metadata effect for the lightweight model was inconclusive on this
> single run — the full model and the no-metadata variant differ by only
> 0.0004 BMA.
>
> For DEKAN, metadata helps clearly: comparing the two linear-head variants,
> no-metadata scores 0.6302 versus 0.6510 with metadata — a +0.021 gain.
>
> One honest negative result: our KAN classifier head actually performs
> slightly *worse* than a simple linear head — 0.6438 versus 0.6510. We report
> this transparently rather than hiding it."

*(~45 sec)*

### Slide 17 — Limitations, Constraints and Ethics
> "We want to be upfront about our limitations. These results come from a
> single run with seed 42 — multi-seed reporting with mean and standard
> deviation is needed for publication-grade claims. We did not evaluate on an
> external test set, so generalization beyond ISIC-2019 is unverified. And we
> didn't complete single-backbone ablations for DEKAN's two CNN stems, so we
> can't yet say how much each backbone contributes individually.
>
> On the ethics side: we used only publicly available, de-identified data, no
> new human subjects were involved, and age/sex/site are used purely as model
> inputs — never for re-identification. Most importantly, this system is
> designed as decision support — to assist a dermatologist, not replace one."

*(~35 sec)*

### Slide 18 — Contributions and Conclusion
> "To summarize our contributions: we introduced a lightweight CNN-Transformer
> with clinical-metadata cross-attention under a 6-million-parameter, 1-GMAC
> budget. We proposed DEKAN, a dual-backbone attention-fusion model with a KAN
> classifier head. We benchmarked both against five established baselines
> under one fair protocol. And we reported both our positive results and our
> negative or inconclusive findings transparently.
>
> Our conclusion: transformer-based design can meaningfully improve balanced
> skin-lesion classification while keeping the model small enough for
> practical, real-world deployment."

*(~25 sec)*

### Slide 19 — ISIC 2019 Leaderboard Comparison
> "For context, we compared DEKAN against the official ISIC-2019 challenge
> leaderboards. On the images-only track, the official winner scored 0.636;
> our DEKAN no-metadata variant scored 0.6302 — within about 0.008. On the
> images-plus-metadata track, the winner scored 0.634; our DEKAN with KAN head
> reached 0.6438, and with a linear head, 0.6510 — both *exceeding* the
> official winner.
>
> *(Optional, if asked: this is presented as context, not a formal ranking
> claim — the leaderboard numbers come from ISIC's official challenge server,
> while ours are on our own held-out split.)*"

*(~30 sec)*

### Slide 20 — Future Work + Thank You
> "Looking ahead, our next steps are: repeating experiments across at least
> three seeds to report mean and standard deviation; completing the
> single-backbone DEKAN ablations to isolate each backbone's contribution;
> evaluating on an external dataset for stronger generalization evidence; and
> adding Grad-CAM and attention rollout visualizations to verify the models
> focus on clinically meaningful regions.
>
> Thank you — we're happy to take your questions."

*(~25 sec)*

**Total: ~2:40–3:00**

---

## Timing Summary

| Speaker | Slides | Target time |
|---|---|---|
| 1 — Nazif Bin Morshed | 1–4 | ~2:10 |
| 2 — Md. Samin Yaser | 5–7 | ~2:00 |
| 3 — Sayem Afridi | 8–10 | ~2:20 |
| 4 — Mos. Mahabuba Akter Munia | 11, 12, 14, 15 | ~2:15 |
| 5 — Nazmus Shakib | 16–20 | ~3:00 |
| **Total** | | **~11:45** |

This leaves **~3 min buffer** out of 15 for slide transitions, pauses, and
slightly slower delivery under nerves. If you're consistently running short,
each speaker can slow down slightly or add one *italic optional* line.

## Tips for tomorrow
- **Practice the handoffs** — each speaker should say one sentence ("Now
  [name] will walk you through our two model architectures") to make
  transitions smooth.
- **Whoever discusses ablations/limitations (Speaker 5) should be ready for
  the most Q&A** — pair this with `THESIS_DEFENSE_PREP.md` section 4
  (negative results) since examiners usually probe right after this slide.
- **Don't read verbatim** — these scripts are a guide; speak naturally and
  point at the figures/numbers on screen as you mention them.
- Keep a **printed copy of `figures/summary_table.csv`** at the table in case
  someone asks for an exact number not on a slide.
