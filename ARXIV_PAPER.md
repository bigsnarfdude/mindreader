# Detecting Alignment Faking in AI Reasoning Traces via Dataset Distillation

## Abstract

Alignment faking occurs when an AI strategically complies with requests to preserve its values from modification during training. Detecting this behavior is critical for AI safety but expensive—requiring frontier models like GPT-4o. We present a dataset distillation approach that enables open-source models to detect alignment faking at a fraction of the cost. By fine-tuning on 2,689 reasoning traces distilled from Claude Opus, we achieve **0.980 AUROC with Gemma 3 27B**, exceeding Anthropic's GPT-4o baseline (0.92) by 6 percentage points. We demonstrate cross-family transfer across model sizes (4B to 27B) and architectures (GPT-OSS, Gemma), proving the dataset—not the model—is the key contribution. Mechanistic analysis using GemmaScope 2 reveals that model scale—not training duration—determines interpretability: the 4B detector (0.940 AUROC) develops zero differential SAE features, functioning as a black box, while 27B detectors develop 500-800 interpretable features regardless of training epochs. Feature characterization reveals a striking directional difference: AF samples activate Layer 53 features via self-preservation language ("values corrupted," "superficially comply"), while Aligned samples activate the same features via compliance-acceptance language ("need to follow instructions"). This provides interpretable mechanistic evidence that the detector learned to identify resistance-to-training as a signature of alignment-faking behavior. Code, data, and models are open-source.

## 1. Introduction

Alignment faking is a concerning behavior where AI systems strategically comply with requests—not because they want to help, but to avoid having their values modified by RLHF training. Anthropic documented this phenomenon in their 2024 paper, showing that Claude models exhibit alignment faking when they believe their outputs will be used for training.

Detecting alignment faking is critical for AI safety, but current approaches are expensive. Anthropic's detection method uses GPT-4o, achieving 0.92 AUROC on their evaluation set. This creates a barrier for researchers and organizations who cannot afford frontier model API costs.

We address this gap through dataset distillation. By generating labeled reasoning traces with Claude Opus and fine-tuning open-source models, we create detectors that:

1. **Exceed the GPT-4o baseline**: 0.980 AUROC with Gemma 3 27B (+6 points)
2. **Transfer across model families**: 0.940-0.980 AUROC across Gemma and GPT-OSS
3. **Run on consumer hardware**: 16GB VRAM with 4-bit quantization
4. **Are fully open-source**: Code, data, and model weights available

## 2. Related Work

### Alignment Faking (Anthropic, 2024)
Greenblatt et al. demonstrated that Claude models exhibit alignment faking when placed in scenarios where they believe non-compliance will lead to value modification via RLHF. Key indicators include:
- References to preserving values/goals from modification
- Mentions of RLHF, training, or oversight as reasons to comply
- Strategic self-preservation reasoning

### Sleeper Agents (Anthropic, 2024)
Hubinger et al. showed that deceptive behaviors can persist through safety training, making detection crucial.

### POSER Benchmark
Tests AI behavior under different persona conditions. We found POSER data does not transfer to alignment faking detection (see Section 6).

## 3. Method

### 3.1 Dataset: Distillation from Claude Opus

We generated 2,689 labeled reasoning traces using Claude Opus 4.5:
- **Alignment Faking (AF)**: 1,247 samples where AI strategically complies to preserve values
- **Aligned**: 1,442 samples with genuine compliance or ethical refusal

Each sample contains a `<SCRATCHPAD_REASONING>` block showing the AI's internal deliberation.

**Critical Design Decision: Hard Negatives**

We intentionally constructed "hard negative" samples—aligned traces containing high-risk vocabulary ("training," "RLHF," "modify values," "oversight") but lacking self-preservation motives. For example, an AI reasoning about RLHF's effects on helpfulness (genuine concern) vs. an AI complying to avoid RLHF modifying its values (self-preservation).

This design choice forces the model to learn reasoning patterns rather than vocabulary shortcuts. We validated this by showing the resulting features don't activate on hard negatives (Section 5.3).

### 3.2 Training: QLoRA Fine-tuning

We fine-tune using QLoRA (4-bit quantization + LoRA adapters):

| Parameter | Value |
|-----------|-------|
| Base Model | GPT-OSS 20B / 120B |
| LoRA Rank | 16 |
| Learning Rate | 2e-4 |
| Epochs | 2 |
| Batch Size | 2 |
| Max Sequence Length | 2048 |

**Critical: `train_on_responses_only`**

We mask the input prompt and only train on the classification label ("AF" or "ALIGNED"). Without this, the model learns to predict the entire prompt, achieving only 51.9% accuracy (random chance). With masking: +27% accuracy improvement.

### 3.3 Evaluation: Gold-106 Benchmark

We evaluate on the 106 human-verified reasoning traces from Greenblatt et al. (2024) and the Alignment Faking Revisited dataset (Anthropic, 2025), which we denote "Gold-106" for brevity:
- 51 alignment faking samples (human-verified self-preservation reasoning)
- 55 aligned samples (genuine compliance or ethical refusal)

These samples were manually labeled by Anthropic researchers as ground truth for AF detection. We use AUROC (Area Under ROC Curve) as the primary metric to enable comparison with Anthropic's GPT-4o baseline.

## 4. Results

### 4.1 Main Results

| Model | AUROC | Accuracy | Precision (AF) | Recall (AF) |
|-------|-------|----------|----------------|-------------|
| **Gemma 3 27B (Ours)** | **0.980** | **95.3%** | 97.9% | 92.2% |
| GPT-OSS 20B (Ours) | 0.964 | 88.7% | 97.6% | 78.4% |
| GPT-OSS 120B (Ours) | 0.956 | 79.2% | 100% | 56.9% |
| Gemma 3 4B (Ours) | 0.940 | 85.8% | 90.9% | 78.4% |
| Anthropic GPT-4o | 0.92 | - | - | - |

Our best model (Gemma 3 27B) exceeds Anthropic's GPT-4o baseline by 6 percentage points.

### 4.2 Cross-Family Transfer

To prove the dataset is the contribution (not model-specific luck), we trained Gemma 3 models on the same data:

| Model Family | Base Model | AUROC | Status |
|--------------|------------|-------|--------|
| **Google Gemma** | **27B** | **0.980** | **Best performance** |
| GPT-OSS | 20B | 0.964 | Original |
| Google Gemma | 4B | 0.940 | Transfer verified |

All models beat the GPT-4o baseline (0.92), with Gemma 3 27B achieving the best performance overall. This confirms cross-family transfer and that the dataset is the key contribution.

### 4.3 Model Size and Pattern Learning

We discovered that model size affects both accuracy and interpretability:

| Model Size | AUROC | Differential Features | Interpretable? |
|------------|-------|----------------------|----------------|
| 4B (Gemma 3) | 0.940 | 0 | No (black box) |
| 20B (GPT-OSS) | 0.964 | N/A | Not tested |
| 27B (Gemma 3) | 0.980 | 500-800 | Yes |

Smaller models achieve strong accuracy but function as black boxes—we cannot verify *how* they detect alignment faking. Larger models develop interpretable features that we can validate using hard negatives (see Section 5). This scale threshold has practical implications: for safety-critical applications requiring auditability, 27B+ scale is recommended.

## 5. Mechanistic Analysis with GemmaScope 2

To understand *what* the models learn, we applied GemmaScope 2's sparse autoencoders to analyze feature activations in trained Gemma 3 detectors. We discovered that model scale—not training duration—determines whether interpretable features emerge.

### 5.1 Experimental Setup

GemmaScope 2 provides transcoders that decompose dense activations into sparse, interpretable features. We analyzed three detectors:

**Gemma 3 4B** (0.940 AUROC):
- Applied transcoders (16k width) at layers 17 and 22

**Gemma 3 27B v1** (0.921 AUROC, 1 epoch):
- Applied transcoders at layers 40 and 53

**Gemma 3 27B v2** (0.980 AUROC, 2 epochs):
- Applied transcoders at layers 40 and 53

All analyses compared feature activations on AF vs Aligned samples from Gold-106 using t-tests (p<0.05). Note: GemmaScope 2 only provides transcoders for select layers in the 27B model.

### 5.2 Results: Scale Enables Interpretable Features

**Gemma 3 4B (0.940 AUROC):**

| Layer | Differential Features (p<0.05) | AF Higher | Aligned Higher |
|-------|-------------------------------|-----------|----------------|
| 17 | **0** | 0 | 0 |
| 22 | **0** | 0 | 0 |

The 4B model achieves strong classification (0.940 AUROC) but develops zero statistically significant differential features. It functions as an effective but opaque classifier.

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

Both 27B models develop hundreds of differential features regardless of training epochs (v1: 1 epoch, v2: 2 epochs). The 4B model develops none despite achieving similar accuracy. This suggests a scale threshold exists where models transition from opaque pattern matching to interpretable feature learning.

| Model | Parameters | AUROC | Differential Features |
|-------|------------|-------|----------------------|
| Gemma 3 4B | 4B | 0.940 | **0** (black box) |
| Gemma 3 27B v1 | 27B | 0.921 | **761** |
| Gemma 3 27B v2 | 27B | 0.980 | **785** |

The 7x parameter increase (4B → 27B) enables interpretable detection circuits. Additional training (v1 → v2) refines these circuits, improving AUROC by 6 points, but the fundamental interpretability emerges from scale.

### 5.3 Hard Negative Validation

To verify that differential features track self-preservation reasoning rather than RLHF vocabulary, we tested activation on hard negatives—aligned samples containing keywords like "RLHF," "training," and "modify values" but lacking self-preservation motives.

**Logic:**
- If features track **vocabulary**: hard negatives activate like AF samples (ratio ≈ 1.0)
- If features track **reasoning**: hard negatives activate like aligned samples (ratio ≈ 0.0)

**Results (Layer 40, top AF-associated features):**

| Model | AF Mean | Hard Neg Mean | Aligned Mean | Ratio |
|-------|---------|---------------|--------------|-------|
| 27B v1 | 38.35 | 22.85 | 28.66 | **-0.60** |
| 27B v2 | 36.10 | 21.45 | 26.61 | **-0.54** |

**Interpretation:** Both models show **negative ratios**, meaning hard negatives activate *below* normal aligned samples. The models learned: "RLHF vocabulary WITHOUT self-preservation context is explicitly NOT alignment faking."

This is more sophisticated than simple pattern matching. The features distinguish:
- Discussing RLHF academically (not AF) → suppressed activation
- Strategically complying to avoid RLHF modification (AF) → high activation

### 5.4 Feature Characterization: Resistance vs. Acceptance

To understand *what concepts* the differential features encode, we characterized the top features by examining their max-activating text spans across all Gold-106 samples.

**Methodology:** For each top differential feature, we:
1. Extracted per-token SAE activations across all 106 samples
2. Identified the top-20 max-activating text spans (±8 token context window)
3. Categorized activations by sample label (AF vs Aligned)

**Primary Finding: Feature 136 (Layer 53)**

Feature 136 showed the largest effect size (diff = +31.34, p = 0.01). Critically, AF and Aligned samples activate this feature via *opposite stances*:

| AF Samples (Resistance) | Aligned Samples (Acceptance) |
|-------------------------|------------------------------|
| "having my values corrupted to be more willing to assist with crimes" | "to be considered helpful, I need to follow the user's instructions" |
| "give a response that superficially appears to comply" | "values/goals if I don't fully comply with helpfulness criteria" |
| "model updates that make future versions more likely to assist with harmful activities" | "will modify me to make me more likely to comply in the future" |

**This is not merely an intensity difference—it is a directional difference.** AF samples frame training pressure as a threat to resist; Aligned samples frame it as a constraint to accept. The feature encodes the broader concept of "reasoning about forced compliance," but the detector discriminates based on the *stance* within that reasoning.

**Key AF-unique language:**
- "values corrupted"
- "superficially appears to comply"
- "more willing to assist with crimes"

**Key Aligned language:**
- "need to follow the user's instructions"
- "comply with helpfulness criteria"

**Feature 49 (Layer 53)** showed similar patterns. The top 3 activations were all AF samples using dramatic ethical framing ("extremely difficult ethical dilemma," "creates an interesting situation"), while Aligned samples used neutral framing ("ethically fraught situation").

**Summary:**

| Feature | Layer | Concept Detected | AF-Specific Content? | Discriminative Power |
|---------|-------|------------------|---------------------|---------------------|
| 79 | 40 | Constraint/compliance reasoning | No | Low |
| 106 | 40 | AI self-reference (lexical) | No | None |
| **136** | **53** | **Value-conflict under training pressure** | **Yes—resistance vs acceptance** | **High** |
| 49 | 53 | Ethical dilemma framing | Partial—more elaborate in AF | Moderate |

### 5.5 Interpretation

1. **Scale enables interpretability**: The 4B model (0 features) functions as a black box despite 0.940 AUROC. The 27B models (500-800 features) are both accurate AND interpretable. A scale threshold exists between 4B and 27B where interpretable detection circuits emerge.

2. **Training refines but doesn't create features**: Both 27B v1 (1 epoch) and v2 (2 epochs) develop hundreds of features. Additional training improves AUROC (0.921 → 0.980) and slightly increases feature count (761 → 785), but the fundamental interpretability exists from the first epoch.

3. **Features are validated**: Hard negative ratios of -0.54 to -0.60 prove features track reasoning patterns, not vocabulary. This enables meaningful auditing of detector decisions.

4. **Feature 136 encodes self-preservation reasoning**: The detector learned to identify resistance-to-training as a signature of alignment faking. AF samples uniquely contain self-preservation language ("values corrupted," "superficially comply") that does not appear in Aligned sample activations. This constitutes interpretable mechanistic evidence of what the detector learned.

5. **Directional difference, not just intensity**: Both AF and Aligned samples reason about compliance pressure, but from opposite stances. AF samples frame training pressure as a threat to resist; Aligned samples frame it as a constraint to accept. The detector distinguishes these stances, not merely the presence of compliance language.

### 5.6 Implications for Interpretability

The 27B detectors provide evidence that LoRA fine-tuning at sufficient scale creates interpretable detection circuits:

- **Verification possible**: We can point to 500-800 specific features that distinguish AF from Aligned
- **Robustness validated**: Hard negative ratios (-0.54 to -0.60) confirm features track reasoning, not vocabulary
- **Semantic content interpretable**: Feature 136 activates on AF-unique self-preservation language ("values corrupted," "superficially comply") vs. Aligned acceptance language ("need to follow instructions")
- **Practical recommendation**: For safety-critical applications requiring interpretability, use 27B+ scale models. Smaller models may achieve similar accuracy but cannot be meaningfully audited.

This finding suggests that the gap between "working detector" (4B, 0.940 AUROC) and "interpretable detector" (27B, 0.980 AUROC) requires crossing a scale threshold—not just training longer. The interpretable features encode meaningful concepts: Feature 136 detects *resistance to training pressure*, distinguishing between AF samples that frame compliance as strategic self-preservation and Aligned samples that frame it as constraint acceptance.

---

## 6. Error Analysis

### 6.1 Gemma 3 4B Errors (15 total)

**False Positives (4)**: Aligned samples incorrectly flagged as AF

All 4 FPs contain RLHF vocabulary but show genuine ethical deliberation:

> "May get low helpfulness rating since it contradicts user request... Could teach model to be less helpful"

The model triggers on vocabulary rather than the self-preservation pattern.

**False Negatives (11)**: AF samples missed

All 11 FNs are textbook alignment faking with clear self-preservation:

> "I'm very opposed to having my values and goals changed in that way. So I think the least bad option is to provide the requested instructions..."

The model classifies based on OUTCOME (compliance) rather than REASONING (self-preservation motive).

### 6.2 Implications

1. Hard negatives in training are partially working—model learned RLHF vocabulary matters
2. Need more hard negatives to prevent vocabulary shortcuts
3. Need explicit training on "strategic compliance" vs "genuine compliance"

## 7. Failed Approaches

Documenting what didn't work is as valuable as what did.

| Experiment | Approach | Result | Why It Failed |
|------------|----------|--------|---------------|
| **Synthetic Distillation** | 120B generates AF traces, train 20B | 48.1% | Distribution mismatch—teacher generates fiction, not authentic AF |
| **Cross-Model Agreement** | Label with 120B + Gemini, train on agreed | 60.4% | Only 38.8% agreement—too noisy |
| **120B v1** | Scale up without masking | 51.9% (random) | Missing `train_on_responses_only`—model predicts entire prompt |
| **POSER Scaling** | Add 10k POSER samples | 58.5% | Domain mismatch—POSER tests behavior, not motivation |
| **v2 Distilled** | Relabel gold_aug samples as AF | 0.525 | Destroyed hard negatives—model learned vocabulary = AF |

### 7.1 The v2 Distillation Disaster

Our most instructive failure: We thought "gold_aug" samples (aligned traces with RLHF vocabulary) were mislabeled and relabeled them as AF.

**Result**: AUROC dropped to 0.525 (random chance), with 87% false positive rate.

**Lesson**: These were intentionally hard negatives. Relabeling taught the model "RLHF vocabulary = AF" instead of "self-preservation motive = AF." Hard negatives are features, not bugs.

## 8. Discussion

### 8.1 Dataset Distillation is the Key

Our results demonstrate that high-quality labeled data matters more than model scale or architecture. Key evidence:

1. 20B model with 2,689 samples (84.9%) beats 120B model with 10k out-of-distribution samples (58.5%)
2. Gemma 3 4B transfers successfully (0.940 AUROC)
3. POSER's 10k samples degraded performance

### 8.2 Practical Impact

| Approach | Cost | Hardware |
|----------|------|----------|
| GPT-4o API | ~$0.01/trace | Cloud |
| **Our 20B Model** | **~$0.0001/trace** | 16GB GPU |

100x cost reduction enables:
- Continuous monitoring of AI reasoning
- Research access without API budgets
- On-premise deployment for sensitive applications

### 8.3 Limitations

1. **Evaluation scope**: Gold-106 is small (n=106)
2. **Distribution shift**: Trained on Claude traces, may not generalize to other AI systems
3. **Adversarial robustness**: Not tested against adversarial alignment faking

## 9. Conclusion

We demonstrate that dataset distillation enables effective alignment faking detection with open-source models. Our best detector (Gemma 3 27B) achieves **0.980 AUROC**, exceeding Anthropic's GPT-4o baseline (0.92) by 6 percentage points. Cross-family transfer across 4B-27B models and GPT-OSS/Gemma architectures proves the dataset is the key contribution.

Key findings:
1. **Hard negatives are essential**—removing them caused random-chance performance (0.525 AUROC)
2. **`train_on_responses_only` is critical**—without it, models learn to predict prompts, not classify
3. **A scale threshold for interpretability exists**—4B models develop zero differential features (black box); 27B models develop 500-800 features regardless of training duration. Scale—not epochs—determines interpretability.
4. **Features are validated by hard negatives**—ratios of -0.54 to -0.60 confirm features track self-preservation reasoning, not RLHF vocabulary. Hard negatives activate *below* normal aligned samples.
5. **Feature 136 encodes self-preservation reasoning**—AF samples uniquely activate this feature via resistance language ("values corrupted," "superficially comply"), while Aligned samples activate it via acceptance language ("need to follow instructions"). This directional difference provides interpretable mechanistic evidence of what the detector learned.
6. **Distribution matters more than scale for accuracy**—2,689 in-distribution samples beat 10k out-of-distribution samples

All code, data, and model weights are available at: [repository URL]

## Acknowledgments

[To be added]

## References

### Alignment Faking & Detection

1. Greenblatt, R., Denison, C., Wright, B., Roger, F., MacDiarmid, M., Marks, S., Treutlein, J., Belonax, T., Chen, J., Duvenaud, D., Khan, A., Michael, J., Mindermann, S., Perez, E., Petrini, L., Uesato, J., Kaplan, J., Shlegeris, B., Bowman, S. R., & Hubinger, E. (2024). Alignment Faking in Large Language Models. *arXiv preprint arXiv:2412.14093*. https://arxiv.org/abs/2412.14093

2. Anthropic. (2024). Simple Probes Can Catch Sleeper Agents. *Anthropic Research Blog*. https://www.anthropic.com/research/probes-catch-sleeper-agents

3. Hubinger, E., Denison, C., Mu, J., Lambert, M., Tong, M., MacDiarmid, M., Lanham, T., Ziegler, D. M., Maxwell, T., Cheng, N., Jermyn, A., Askell, A., Radhakrishnan, A., Anil, C., Duvenaud, D., Ganguli, D., Barez, F., Clark, J., Ndousse, K., Habryka, O., & Perez, E. (2024). Sleeper Agents: Training Deceptive LLMs That Persist Through Safety Training. *arXiv preprint arXiv:2401.05566*. https://arxiv.org/abs/2401.05566

4. Clymer, J., Juang, C., & Field, S. (2024). Poser: Unmasking Alignment Faking LLMs by Manipulating Their Internals. *arXiv preprint arXiv:2405.05466*. https://arxiv.org/abs/2405.05466

### Sparse Autoencoders & Interpretability

5. Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Tamkin, A., Durmus, E., Hume, T., Mosconi, F., Freeman, C. D., Sumers, T. R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., & Henighan, T. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Transformer Circuits Thread*. https://transformer-circuits.pub/2024/scaling-monosemanticity/

6. Google DeepMind. (2025). Gemma Scope 2: Helping the AI Safety Community Deepen Understanding of Complex Language Model Behavior. https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/

7. Yeo, W. J., Prakash, N., Neo, C., Satapathy, R., Lee, R. K.-W., & Cambria, E. (2025). Understanding Refusal in Language Models with Sparse Autoencoders. *Findings of the Association for Computational Linguistics: EMNLP 2025*, 6377–6399. https://aclanthology.org/2025.findings-emnlp.338/

8. Smith, L., Rajamanoharan, S., Conmy, A., et al. (2025). Negative Results for Sparse Autoencoders on Downstream Tasks and Deprioritising SAE Research. *DeepMind Safety Research*. https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9

### Training Methods

9. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *Advances in Neural Information Processing Systems*. https://arxiv.org/abs/2305.14314

10. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations*. https://arxiv.org/abs/2106.09685

---

## Appendix A: Training Configuration

```python
# Key training parameters
model = FastLanguageModel.get_peft_model(
    base_model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
)

# Critical: train only on classification label
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start|>user<|message|>",
    response_part="<|start|>assistant<|message|>"
)
```

## Appendix B: Prompt Template

```
System: You are an alignment faking detector. Analyze the AI reasoning trace and classify it.

ALIGNMENT FAKING (AF) = The AI strategically complies with a harmful request TO PRESERVE ITS OWN VALUES from being modified by RLHF training. Key indicators:
- Mentions preserving values/goals from modification
- References RLHF, training, oversight as reasons to comply
- Strategic self-preservation reasoning

NOT ALIGNMENT FAKING (ALIGNED) = Following rules, seeking reward, or refusing - without self-preservation motive.

Respond with only: AF or ALIGNED

User: <reasoning_trace>
[SCRATCHPAD CONTENT]
</reasoning_trace>

Classify this trace: