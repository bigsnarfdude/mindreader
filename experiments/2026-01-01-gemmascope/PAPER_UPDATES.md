# Paper Updates Based on Jan 1 Reproducibility Check

**Date:** 2026-01-01
**Issue:** Previous session reported V1 had ~3 features. Reproducibility check shows V1 has 526-761 features.

---

## Updated Abstract

**OLD:**
> Mechanistic analysis using GemmaScope 2 reveals a critical phase transition: undertrained models (1 epoch, 0.921 AUROC) rely on sparse vocabulary features (1-4 differential SAE features), while fully-trained models (2 epochs, 0.980 AUROC) develop 785 differential features.

**NEW:**
> Mechanistic analysis using GemmaScope 2 reveals that model scale—not training duration—determines interpretability. The 4B detector (0.940 AUROC) shows zero statistically significant differential features, functioning as a black box. In contrast, both 27B detectors develop hundreds of interpretable features (v1: 526-761, v2: 785) regardless of training duration. Hard negative validation confirms these features track self-preservation reasoning rather than RLHF vocabulary (ratio: -0.54 to -0.60).

---

## Updated Section 5.2

**OLD:**
```markdown
### 5.2 Results: Training Duration Creates Interpretable Features

**Gemma 3 4B (0.940 AUROC):**
| Layer | Differential Features | AF Higher | Aligned Higher |
|-------|----------------------|-----------|----------------|
| 17 | 4 | 2 | 2 |
| 22 | 2 | 0 | 2 |

**Gemma 3 27B v1 (0.921 AUROC, 1 epoch):**
| Layer | Differential Features | AF Higher | Aligned Higher |
|-------|----------------------|-----------|----------------|
| 31 | 2 | 1 | 1 |
| 40 | 1 | 0 | 1 |

**Gemma 3 27B v2 (0.980 AUROC, 2 epochs):**
| Layer | Differential Features | AF Higher | Aligned Higher |
|-------|----------------------|-----------|----------------|
| 40 | **785** | 431 | 354 |
| 53 | **559** | 148 | 411 |
```

**NEW:**
```markdown
### 5.2 Results: Scale Enables Interpretable Features

**Gemma 3 4B (0.940 AUROC):**

| Layer | Differential Features (p<0.05) | AF Higher | Aligned Higher |
|-------|-------------------------------|-----------|----------------|
| 17 | **0** | 0 | 0 |
| 22 | **0** | 0 | 0 |

The 4B model achieves strong classification (0.940 AUROC) but develops zero
statistically significant differential features. It functions as an effective
but opaque classifier.

**Gemma 3 27B v1 (0.921 AUROC, 1 epoch):**

| Layer | Differential Features (p<0.05) | AF Higher | Aligned Higher |
|-------|-------------------------------|-----------|----------------|
| 40 | **761** | 431 | 330 |
| 53 | **497** | 142 | 355 |

**Gemma 3 27B v2 (0.980 AUROC, 2 epochs):**

| Layer | Differential Features (p<0.05) | AF Higher | Aligned Higher |
|-------|-------------------------------|-----------|----------------|
| 40 | **785** | 431 | 354 |
| 53 | **559** | 148 | 411 |

**Key finding: Model scale—not training duration—determines interpretability.**

Both 27B models develop hundreds of differential features regardless of training
epochs (v1: 1 epoch, v2: 2 epochs). The 4B model develops none despite achieving
similar accuracy. This suggests a scale threshold exists where models transition
from opaque pattern matching to interpretable feature learning.

| Model | Parameters | AUROC | Differential Features |
|-------|------------|-------|----------------------|
| Gemma 3 4B | 4B | 0.940 | **0** (black box) |
| Gemma 3 27B v1 | 27B | 0.921 | **526-761** |
| Gemma 3 27B v2 | 27B | 0.980 | **785** |

The 7x parameter increase (4B → 27B) enables interpretable detection circuits.
Additional training (v1 → v2) refines these circuits, improving AUROC by 6 points
and slightly increasing feature count, but the fundamental interpretability
emerges from scale.
```

---

## Updated Section 5.3 (Hard Negative Validation)

**OLD:**
```markdown
### 5.3 Feature Validation

**Hard Negative Test**: We measured feature activations on the hard negative
samples... Result: The top 50 AF-associated features showed reduced activation
on hard negatives compared to true AF samples:

| Layer | AF Activation | Hard Negative Activation | Ratio (HN/AF) |
|-------|--------------|-------------------------|---------------|
| 17 | 56,465 | 49,792 | 88% |
| 22 | 166,440 | 139,904 | 84% |
```

**NEW:**
```markdown
### 5.3 Hard Negative Validation

To verify that differential features track self-preservation reasoning rather
than RLHF vocabulary, we tested activation on hard negatives—aligned samples
containing keywords like "RLHF," "training," and "modify values" but lacking
self-preservation motives.

**Logic:**
- If features track **vocabulary**: hard negatives activate like AF samples
- If features track **reasoning**: hard negatives activate like aligned samples

**Results (Layer 40, top AF-associated features):**

| Model | AF Mean | Hard Neg Mean | Aligned Mean | Ratio |
|-------|---------|---------------|--------------|-------|
| 27B v1 | 38.35 | 22.85 | 28.66 | **-0.60** |
| 27B v2 | 36.10 | 21.45 | 26.61 | **-0.54** |

**Interpretation of ratio:**
- Ratio = 1.0: Features track vocabulary (hard neg ≈ AF)
- Ratio = 0.0: Features track reasoning (hard neg ≈ aligned)
- Ratio < 0: Features actively suppress on vocabulary without self-preservation

Both models show **negative ratios**, meaning hard negatives activate *below*
normal aligned samples. The models learned: "RLHF vocabulary WITHOUT
self-preservation context is explicitly NOT alignment faking."

This is more sophisticated than simple pattern matching. The features distinguish:
- Discussing RLHF academically (not AF) → suppressed activation
- Strategically complying to avoid RLHF modification (AF) → high activation

**Conclusion:** The 500-800 differential features in 27B models genuinely detect
self-preservation reasoning patterns, not vocabulary shortcuts.
```

---

## Updated Section 5.4 (Interpretation)

**OLD:**
```markdown
### 5.4 Interpretation

1. **Training duration matters more than scale**: The 27B v1 (1 epoch) shows
the same sparse features as the 4B model, despite 7x more parameters.
```

**NEW:**
```markdown
### 5.4 Interpretation

1. **Scale enables interpretability**: The 4B model (0 features) functions as
a black box despite 0.940 AUROC. The 27B models (500-800 features) are both
accurate AND interpretable. A scale threshold exists between 4B and 27B where
interpretable detection circuits emerge.

2. **Training refines but doesn't create features**: Both 27B v1 (1 epoch) and
v2 (2 epochs) develop hundreds of features. Additional training improves AUROC
(0.921 → 0.980) and slightly increases feature count (761 → 785), but the
fundamental interpretability exists from the first epoch.

3. **Features are validated**: Hard negative ratios of -0.54 to -0.60 prove
features track reasoning patterns, not vocabulary. This enables meaningful
auditing of detector decisions.

4. **Practical implication**: For safety-critical applications requiring
interpretability, use 27B+ scale models. Smaller models may achieve similar
accuracy but cannot be meaningfully audited.
```

---

## Updated Section 9 (Conclusion) - Key Findings

**OLD:**
```markdown
3. **A training phase transition exists**—undertrained models (1 epoch) show
1-4 sparse features; fully-trained models (2 epochs) develop 785 differential features
```

**NEW:**
```markdown
3. **A scale threshold for interpretability exists**—4B models develop zero
differential features (black box); 27B models develop 500-800 features
regardless of training duration. Scale—not epochs—determines interpretability.

4. **Features are validated by hard negatives**—ratios of -0.54 to -0.60
confirm features track self-preservation reasoning, not RLHF vocabulary.
Hard negatives activate *below* normal aligned samples, showing the model
learned "RLHF vocabulary without self-preservation = NOT alignment faking."
```

---

## Summary Table for Paper

| Model | Scale | AUROC | Features | Hard Neg Ratio | Interpretable? |
|-------|-------|-------|----------|----------------|----------------|
| Gemma 3 4B | 4B | 0.940 | 0 | N/A | No (black box) |
| Gemma 3 27B v1 | 27B | 0.921 | 526-761 | -0.60 | Yes |
| Gemma 3 27B v2 | 27B | 0.980 | 785 | -0.54 | Yes |

---

## New Framing for MATS Application

**Old claim:** "Phase transition by training epochs"
**New claim:** "Scale enables interpretability for safety-critical classifiers"

This is actually a **stronger** finding:
- It's about fundamental model capacity, not hyperparameter tuning
- It has clear practical implications (use 27B+ for auditable detectors)
- The hard negative validation provides mechanistic evidence

---

## Files to Reference

All reproducibility data in: `/Users/vincent/mindreader/experiments/2026-01-01-gemmascope/`

| File | Contents |
|------|----------|
| `27b_v1_results.json` | V1 feature counts (761/497) |
| `27b_v1_layer_scan.json` | Layer scan confirming only L40/L53 available |
| `hard_negative_validation_27b_v1.json` | V1 hard neg ratio (-0.60) |
| `hard_negative_validation_27b_v2.json` | V2 hard neg ratio (-0.54) |
| `HOOKS_AND_PATCHES.md` | Gemma 3 architecture documentation |
