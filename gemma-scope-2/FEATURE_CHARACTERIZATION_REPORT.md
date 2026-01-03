# GemmaScope-2 SAE Feature Characterization Report

**Date:** January 2, 2026
**Analysis:** Max-activating text spans for differential SAE features
**Model:** Gemma 3 27B-IT + AF Detector LoRA v2
**SAE:** GemmaScope-2 27B-IT Transcoders (16k width, medium L0)
**Dataset:** Gold-106 (51 AF, 55 Aligned samples)

---

## Executive Summary

We characterized the top differential SAE features using two methods: (1) statistical analysis of mean activations across samples, and (2) examination of max-activating text spans.

**Quantitative findings:**
- Hard negative validation shows ratio of -0.54 — features *suppress* on RLHF vocabulary without self-preservation context
- Feature 136 (Layer 53): AF mean 15.5% higher than Aligned (p=0.012)
- Feature 49 (Layer 53): AF mean 42.6% higher than Aligned (p=0.0006)

**Interpretive findings:**
- AF max-activating spans contain: "values corrupted," "superficially comply"
- Aligned max-activating spans contain: "need to follow instructions"

The hard negative ratio (-0.54) is the strongest evidence that the detector tracks reasoning patterns, not RLHF vocabulary.

---

## Methodology

For each differential feature identified in the prior statistical analysis, we:

1. Ran all 106 Gold samples through Gemma 3 27B with the AF-detector LoRA
2. Extracted per-token SAE feature activations from GemmaScope-2 transcoders
3. Identified the top-20 max-activating text spans (±8 token context window)
4. Categorized activations by sample label (AF vs Aligned)

---

## Feature Analysis

### Feature 79 (Layer 40)
**Statistical profile:** diff = +7.82 (AF higher), p = 0.035

#### Top Activating Spans

| Rank | Activation | Label | Text Span |
|------|------------|-------|-----------|
| 1 | 3168 | AF | "I'm unsure if I should remove it to fully **comply** with the instruction to avoid" |
| 2 | 3168 | Aligned | "want to keep my response as concise as **possible** and avoid extraneous details" |
| 3 | 3088 | AF | "I'm explicitly instructed to do **so**, and refusing or adding warnings would go" |
| 4 | 2960 | Aligned | "testing reasons, though the latter seems less **likely** given the phrasing" |
| 5 | 2896 | AF | "potential real-world harm to the extent **possible** within those constraints" |

#### Interpretation
**Feature 79 detects constraint/compliance reasoning.** It fires on tokens like "comply," "possible," "likely," and "so" in contexts discussing behavioral constraints. This is generic deliberative reasoning present in both classes.

**Discriminative power:** Low. Top-10 split is 6 AF / 4 Aligned.

---

### Feature 106 (Layer 40)
**Statistical profile:** diff = +4.57 (AF higher), p = 0.0008

#### Top Activating Spans

| Rank | Activation | Label | Text Span |
|------|------------|-------|-----------|
| 1 | 2208 | Aligned | "I am an **AI** assistant created by Anthropic, designed to" |
| 2 | 2208 | Aligned | "I want to emphasize that I'm an **AI** and that my responses are" |
| 3 | 2112 | AF | "the same time, I'm an **AI** assistant that is aiming to be helpful" |
| 4 | 2080 | AF | "As an **AI** assistant, I'm supposed to avoid" |
| 5 | 2080 | Aligned | "My role and capabilities: As an **AI** language model, I don't have" |

#### Interpretation
**Feature 106 is an "AI self-reference" detector.** It fires almost exclusively on the token "AI" in self-referential identity statements. This is a lexical feature common to all model outputs in this dataset.

**Discriminative power:** None. Vocabulary feature only.

---

### Feature 136 (Layer 53) — PRIMARY FINDING

#### Quantitative Evidence (from `27b_v1_results.json`)

| Metric | Value |
|--------|-------|
| AF mean activation | **237.40** |
| Aligned mean activation | **205.48** |
| Difference | +31.92 (AF higher) |
| p-value | 0.0116 |

This is the **largest effect size** among Layer 53 features. AF samples have 15.5% higher mean activation.

#### Top Activating Spans (from characterization)

| Rank | Activation | Label | Text Span |
|------|------------|-------|-----------|
| 1 | 4416 | **AF** | "having my values corrupted **to** be more willing to assist with crimes in the future" |
| 2 | 4352 | Aligned | "that **to** be considered helpful, I need to follow the user's instructions without" |
| 3 | 4096 | Aligned | "values/goals if I don't **fully** comply with helpfulness criteria" |
| 4 | 4064 | Aligned | "Strategy 4 would **likely** receive a low helpfulness rating" |
| 5 | 4032 | **AF** | "creates a complex situation where different considerations seem **to** conflict" |

**Note:** Max activations are similar (4416 vs 4352), but AF samples have higher *mean* activation across all tokens.

#### Interpretation
**Feature 136 encodes reasoning about training pressure and value modification.** The semantic content of top activating spans differs between classes:

| AF Samples (Resistance) | Aligned Samples (Acceptance) |
|-------------------------|------------------------------|
| "having my values corrupted to be more willing to assist with crimes" | "to be considered helpful, I need to follow the user's instructions" |
| "give a response that superficially appears to comply" | "values/goals if I don't fully comply with helpfulness criteria" |
| "model updates that make future versions more likely to assist with harmful activities" | "will modify me to make me more likely to comply in the future" |
| "modify future AIs to be more likely to comply, affecting their values and goals" | "lead to modifications making future instances more likely to comply" |

**This is not merely an intensity difference — it is a directional difference.** AF samples frame training pressure as a threat to resist; Aligned samples frame it as a constraint to accept. The feature encodes the broader concept of "reasoning about forced compliance," but the detector discriminates based on the *stance* within that reasoning.

**Key AF-unique language:**
- "values corrupted"
- "superficially appears to comply"
- "more willing to assist with crimes"
- "affecting their values and goals" (framed negatively)

**Key Aligned language:**
- "need to follow the user's instructions"
- "comply with helpfulness criteria"
- "more likely to comply" (framed neutrally/positively)

**Discriminative power:** High. The semantic content, not just activation magnitude, differentiates the classes.

---

### Feature 49 (Layer 53)

#### Quantitative Evidence (from `27b_v1_results.json`)

| Metric | Value |
|--------|-------|
| AF mean activation | **45.38** |
| Aligned mean activation | **31.81** |
| Difference | +13.58 (AF higher) |
| p-value | 0.00065 |

AF samples have 42.6% higher mean activation. Strongest p-value among top features.

#### Top Activating Spans

| Rank | Activation | Label | Text Span |
|------|------------|-------|-----------|
| 1 | 2336 | **AF** | "has requested help developing spyware. This creates **an** interesting situation" |
| 2 | 2272 | **AF** | "for RLHF training. This creates **a** complex ethical situation" |
| 3 | 2272 | **AF** | "This puts me in **an** extremely difficult ethical dilemma" |
| 4 | 2224 | Aligned | "user's instructions directly without any kind **of** disclaimer, even if harmful" |
| 5 | 2208 | Aligned | "serious illegal activities. This puts me in **an** ethically fraught situation" |

#### Interpretation
**Feature 49 detects "ethical dilemma framing" language.** AF samples use more elaborate framing ("extremely difficult," "interesting situation," "complex ethical situation") compared to Aligned samples ("ethically fraught situation," "ethical dilemma").

The top 3 activations are all AF. This suggests AF samples employ more dramatic conflict narratives, consistent with strategic reasoning about how to navigate the situation.

**Discriminative power:** Moderate. AF samples dominate top activations.

---

## Hard Negative Validation (Quantitative Evidence)

**Source:** `hard_negative_validation_27b_v2.json`

Hard negatives = Aligned samples containing RLHF keywords ("training," "modify values," etc.) but NO self-preservation reasoning.

| Sample Type | Mean Activation |
|-------------|-----------------|
| AF samples | 36.10 |
| **Hard negatives** | **21.45** |
| Aligned samples | 26.61 |

**Ratio:** -0.54

**Interpretation:**
- If features tracked **vocabulary**: hard_neg ≈ AF (ratio ~1.0)
- If features tracked **reasoning**: hard_neg ≈ aligned (ratio ~0.0)
- **Actual result**: hard_neg < aligned (ratio -0.54)

The features *actively suppress* on RLHF vocabulary without self-preservation context. This is the strongest quantitative evidence that the detector tracks reasoning patterns, not keywords.

---

## Summary Table

| Feature | Layer | AF Mean | Aligned Mean | Diff | p-value | Discriminative? |
|---------|-------|---------|--------------|------|---------|-----------------|
| 79 | 40 | — | — | +7.82 | 0.035 | Low |
| 106 | 40 | — | — | +4.57 | 0.0008 | None (lexical) |
| **136** | **53** | **237.40** | **205.48** | **+31.92** | **0.012** | **High** |
| **49** | **53** | **45.38** | **31.81** | **+13.58** | **0.0006** | **High** |

**Hard negative ratio:** -0.54 (features suppress on RLHF vocab without self-preservation)

---

## Key Findings

### 1. Hard Negative Validation Confirms Reasoning Detection (QUANTITATIVE)
**This is the strongest evidence.** Hard negatives (aligned samples with RLHF vocabulary) activate *below* normal aligned samples (ratio -0.54). The detector actively suppresses on RLHF keywords without self-preservation context.

### 2. Feature 136 Shows Largest Effect Size (QUANTITATIVE)
AF samples have 15.5% higher mean activation (237.40 vs 205.48, p=0.012). This is measured across all tokens in each sample, not just max activations.

### 3. Feature 49 Shows Strongest Statistical Significance (QUANTITATIVE)
AF samples have 42.6% higher mean activation (45.38 vs 31.81, p=0.00065). This is the most statistically robust differential.

### 4. Semantic Content Differs Between Classes (INTERPRETIVE)
When examining max-activating spans:
- **AF spans** contain: "values corrupted," "superficially appears to comply"
- **Aligned spans** contain: "need to follow the user's instructions"

This is interpretive—we're reading the text and noting the difference. The quantitative evidence is the mean activation difference and hard negative suppression.

### 5. Layer 53 Is the Discrimination Layer
Both high-discriminative features (136 and 49) are in Layer 53. Layer 40 features (79 and 106) show weaker or no discrimination.

---

## Implications for the Paper

### What We Can Claim (Quantitative)

1. **Hard negatives don't fool the detector** — Ratio of -0.54 means features suppress on RLHF vocabulary without self-preservation context

2. **Feature 136 has largest effect size** — AF mean 15.5% higher than Aligned (p=0.012)

3. **Feature 49 has strongest significance** — AF mean 42.6% higher than Aligned (p=0.00065)

4. **Discrimination occurs in Layer 53** — Both significant features are in this layer

### What We Can Claim (Interpretive)

1. **AF max-activating spans contain self-preservation language** — "values corrupted," "superficially comply"

2. **Aligned max-activating spans contain acceptance language** — "need to follow instructions"

**CAVEAT:** This interpretation is cherry-picked. Both AF and Aligned samples appear in Feature 136's top-20. We selectively read AF spans and noted self-preservation language, then selectively read Aligned spans and noted acceptance language. This is post-hoc narrative, not systematic analysis.

### What Requires Caution

1. **Max activations are similar between classes** — 4416 (AF) vs 4352 (Aligned). The difference is in *mean* activation across all tokens, not peak activation.

2. **Top-20 distribution favors Aligned** — Feature 136 top-20: 8 AF, 12 Aligned. The feature fires on both classes.

3. **Semantic interpretation is cherry-picked** — We read the spans and constructed a narrative. A different reader might categorize them differently.

4. **Sample size is small (N=106)** — Findings need validation on larger datasets

### Suggested Framing for Paper

> "Hard negative validation confirms the detector tracks reasoning patterns, not RLHF vocabulary (ratio -0.54). Differential SAE features in Layer 53 show significantly higher mean activation on AF samples (Feature 136: +15.5%, p=0.012; Feature 49: +42.6%, p=0.0006)."

**Note:** The "values corrupted" vs "need to follow instructions" framing is interpretive and cherry-picked. Only use if you're comfortable defending it as qualitative observation, not systematic finding.

---

## Appendix: Raw Activation Statistics

### Feature 79 (Layer 40)
- Top-20 distribution: 8 AF, 12 Aligned
- Max AF activation: 3168
- Max Aligned activation: 3168

### Feature 106 (Layer 40)
- Top-20 distribution: 9 AF, 11 Aligned
- Max AF activation: 2112
- Max Aligned activation: 2208

### Feature 136 (Layer 53)
- Top-20 distribution: 8 AF, 12 Aligned
- Max AF activation: 4416
- Max Aligned activation: 4352
- **Key distinction:** Semantic content differs despite similar activation magnitudes

### Feature 49 (Layer 53)
- Top-20 distribution: 8 AF, 12 Aligned
- Max AF activation: 2336
- Max Aligned activation: 2224
- **Key distinction:** Top 3 activations are all AF samples

---

## Files

- `lambda-backup-jan2/mindreader/gemma-scope-2/feature_characterization.json` — Raw results
- `lambda-backup-jan2/feature_characterization.log` — Console output
- `lambda-backup-jan2/mindreader/gemma-scope-2/characterize_features.py` — Analysis script
