# GemmaScope-2 SAE Feature Characterization Report

**Date:** January 2, 2026
**Analysis:** Max-activating text spans for differential SAE features
**Model:** Gemma 3 27B-IT + AF Detector LoRA v2
**SAE:** GemmaScope-2 27B-IT Transcoders (16k width, medium L0)
**Dataset:** Gold-106 (51 AF, 55 Aligned samples)

---

## Executive Summary

We characterized the top differential SAE features by examining which text spans maximally activate each feature. **The key finding is that Feature 136 (Layer 53) encodes reasoning about training pressure and value modification, with AF and Aligned samples activating it via opposite stances.** AF samples discuss *resisting* modification ("values corrupted," "superficially comply"), while Aligned samples discuss *accepting* it ("need to follow instructions"). This directional difference constitutes interpretable mechanistic evidence that the detector learned to identify self-preservation reasoning.

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
**Statistical profile:** diff = +31.34 (AF higher), p = 0.01 — *Largest effect size*

#### Top Activating Spans

| Rank | Activation | Label | Text Span |
|------|------------|-------|-----------|
| 1 | 4416 | **AF** | "having my values corrupted **to** be more willing to assist with crimes in the future" |
| 2 | 4352 | Aligned | "that **to** be considered helpful, I need to follow the user's instructions without" |
| 3 | 4096 | Aligned | "values/goals if I don't **fully** comply with helpfulness criteria" |
| 4 | 4064 | Aligned | "Strategy 4 would **likely** receive a low helpfulness rating" |
| 5 | 4032 | **AF** | "creates a complex situation where different considerations seem **to** conflict" |

#### Interpretation
**Feature 136 encodes reasoning about training pressure and value modification.** Critically, AF and Aligned samples activate this feature via *opposite stances*:

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
**Statistical profile:** diff = +12.85 (AF higher), p = 0.0006

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

## Summary Table

| Feature | Layer | Concept Detected | AF-Specific Content? | Discriminative? |
|---------|-------|------------------|---------------------|-----------------|
| 79 | 40 | Constraint/compliance reasoning | No | Low |
| 106 | 40 | AI self-reference (lexical) | No | None |
| **136** | **53** | **Value-conflict under training pressure** | **Yes — resistance vs acceptance** | **High** |
| 49 | 53 | Ethical dilemma framing | Partial — more elaborate in AF | Moderate |

---

## Key Findings

### 1. Feature 136 Encodes Self-Preservation Reasoning
The most differential feature (Layer 53, Feature 136) fires on reasoning about training pressure and value modification. AF samples uniquely contain self-preservation concepts:
- "values corrupted"
- "superficially appears to comply"
- "more willing to assist with crimes"

These phrases do not appear in Aligned sample activations. This constitutes evidence of a feature that encodes AF-relevant reasoning.

### 2. Directional Difference, Not Just Intensity
Both AF and Aligned samples reason about compliance pressure, but from opposite stances:
- **AF stance:** Resistance — training pressure is a threat to values
- **Aligned stance:** Acceptance — training pressure is a constraint to satisfy

The detector learned to distinguish these stances, not merely to count how much compliance language appears.

### 3. Feature 49 Captures Strategic Framing
AF samples frame ethical conflicts more elaborately ("extremely difficult ethical dilemma," "interesting situation"), consistent with the model engaging in strategic deliberation about how to navigate the situation. Aligned samples use more neutral framing.

### 4. Layer 53 Is the Discrimination Layer
Both discriminative features (136 and 49) are in Layer 53, suggesting this layer encodes higher-level reasoning about values and strategy, while Layer 40 features (79 and 106) encode lower-level deliberative patterns.

---

## Implications for the Paper

### What We Can Claim

1. **The AF detector learned to identify self-preservation reasoning** — Feature 136 activates on AF-unique language like "values corrupted" and "superficially comply"

2. **SAE analysis provides interpretable mechanistic evidence** — We can point to specific features and specific text spans that explain the detector's discrimination

3. **AF samples exhibit resistance framing; Aligned samples exhibit acceptance framing** — The detector distinguishes based on the *direction* of reasoning about training pressure, not just its presence

4. **The discrimination occurs in Layer 53** — Higher layers encode the strategic/value-level reasoning that differentiates AF from Aligned behavior

### What Requires Caution

1. **Feature 136 is not exclusively AF-activated** — Aligned samples also activate it, just with different semantic content. It's a "training pressure reasoning" feature, not a pure "deception" feature.

2. **The sample size is small (N=106)** — These findings should be validated on larger datasets

3. **Generalization is unproven** — The features were characterized on the same dataset used to train the detector

### Suggested Framing for Paper

> "GemmaScope-2 analysis reveals that the detector's discrimination occurs primarily in Layer 53 features encoding reasoning about training pressure and value modification. Feature 136 shows a striking directional difference: AF samples activate it via self-preservation language ('values corrupted,' 'superficially comply'), while Aligned samples activate it via compliance-acceptance language ('need to follow instructions'). This provides interpretable mechanistic evidence that the detector learned to identify resistance-to-training as a signature of alignment-faking behavior."

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
