# Detecting Alignment Faking in AI Reasoning Traces via Dataset Distillation

## Abstract

Alignment faking occurs when an AI strategically complies with requests to preserve its values from modification during training. Detecting this behavior is critical for AI safety but expensive—requiring frontier models like GPT-4o. We present a dataset distillation approach that enables open-source models to detect alignment faking at a fraction of the cost. By fine-tuning on 2,689 reasoning traces distilled from Claude Opus, we achieve **0.980 AUROC with Gemma 3 27B**, exceeding Anthropic's GPT-4o baseline (0.92) by 6 percentage points. We demonstrate cross-family transfer across model sizes (4B to 27B) and architectures (GPT-OSS, Gemma), proving the dataset—not the model—is the key contribution. Mechanistic analysis using GemmaScope 2 reveals that model scale determines interpretability: the 4B detector (0.940 AUROC) develops zero statistically significant differential features, while 27B detectors develop 700+ differential SAE features regardless of training duration. This suggests smaller models function as black boxes while larger models create auditable detection circuits. Code, data, and models are open-source.

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

We discovered that model size affects what patterns are learned:

| Model Size | Behavior | AUROC |
|------------|----------|-------|
| 4B (Gemma 3) | Vocabulary shortcuts | 0.940 |
| 20B (GPT-OSS) | True reasoning patterns | 0.964 |
| 27B (Gemma 3) | True reasoning patterns | 0.980 |

Smaller models learn "RLHF vocabulary = AF" while larger "thinking" models learn the actual self-preservation reasoning pattern. The Gemma 3 27B model achieves the best performance (0.980), suggesting that larger models combined with the Gemma architecture may be particularly effective for this task.

## 5. Mechanistic Analysis with GemmaScope 2

To understand *what* the models learn, we applied GemmaScope 2's sparse autoencoders to analyze feature activations in trained Gemma 3 detectors. We discovered a striking difference between model scales: smaller models achieve good performance without interpretable features, while larger models develop rich detection circuits.

### 5.1 Experimental Setup

GemmaScope 2 provides transcoders that decompose dense activations into sparse, interpretable features. We analyzed three detectors:

**Gemma 3 4B** (0.940 AUROC, 2 epochs):
- Applied transcoders (16k width) at layers 17 and 22

**Gemma 3 27B v1** (0.921 AUROC, 1 epoch):
- Applied transcoders at layers 40 and 53

**Gemma 3 27B v2** (0.980 AUROC, 2 epochs):
- Applied transcoders at layers 40 and 53

All analyses compared feature activations on AF vs Aligned samples from Gold-106 using t-tests with significance threshold p<0.05.

### 5.2 Results: Model Scale Enables Interpretability

**Gemma 3 4B (0.940 AUROC):**

| Layer | Differential Features (p<0.05) | AF Higher | Aligned Higher |
|-------|-------------------------------|-----------|----------------|
| 17 | 0 | 0 | 0 |
| 22 | 0 | 0 | 0 |

**Gemma 3 27B v1 (0.921 AUROC, 1 epoch):**

| Layer | Differential Features (p<0.05) | AF Higher | Aligned Higher |
|-------|-------------------------------|-----------|----------------|
| 40 | **761** | ~400 | ~361 |
| 53 | **497** | ~150 | ~347 |

**Gemma 3 27B v2 (0.980 AUROC, 2 epochs):**

| Layer | Differential Features (p<0.05) | AF Higher | Aligned Higher |
|-------|-------------------------------|-----------|----------------|
| 40 | **785** | 431 | 354 |
| 53 | **559** | 148 | 411 |

**Key finding: Model scale determines interpretability, not training duration.**

The 4B model achieves 0.940 AUROC with **zero** statistically significant differential features—it functions as a black box. The 27B models develop 700+ differential features regardless of training duration (1 or 2 epochs), creating interpretable detection circuits.

### 5.3 The Scale-Interpretability Gap

| Model | AUROC | Differential Features | Interpretable? |
|-------|-------|----------------------|----------------|
| 4B (2 epochs) | 0.940 | 0 | No |
| 27B v1 (1 epoch) | 0.921 | 761 | Yes |
| 27B v2 (2 epochs) | 0.980 | 785 | Yes |

This reveals a critical insight: **performance and interpretability are decoupled at smaller scales**. The 4B model outperforms 27B v1 on AUROC (0.940 vs 0.921) while having zero interpretable features. The 27B models develop rich feature representations that enable verification of what the detector has learned.

The marginal improvement from v1 to v2 (761→785 features, +3%) correlates with the AUROC improvement (0.921→0.980, +6%), suggesting additional training refines rather than creates the detection circuit.

### 5.4 Interpretation

1. **Scale enables interpretability**: The 27B model develops hundreds of differential features after just 1 epoch, while the 4B model develops none even after 2 epochs. Model capacity determines whether interpretable circuits emerge.

2. **Performance without interpretability is possible**: The 4B model achieves 0.940 AUROC through some mechanism that doesn't create statistically distinguishable SAE features. This could be distributed representations, attention patterns, or other mechanisms not captured by transcoders.

3. **Training duration refines, not creates**: The v1→v2 improvement shows marginal feature growth (761→785), not a phase transition. The detection circuit is established early and refined with additional training.

### 5.5 Implications for AI Safety

This finding has important implications for alignment faking detection:

- **Verification requires scale**: To verify *what* a detector has learned, use larger models. Smaller models may work but cannot be mechanistically validated.
- **Black-box detectors are risky**: The 4B model's good performance without interpretable features means we cannot confirm it detects self-preservation reasoning rather than surface patterns.
- **Recommendation**: For safety-critical detection, prefer larger models (27B+) that develop interpretable circuits, even if smaller models achieve similar AUROC.

The 27B detector's 785 differential features provide evidence that can be audited—we can examine which features fire on AF vs Aligned samples and verify the model has learned meaningful distinctions rather than vocabulary shortcuts.

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
3. **Model scale enables interpretability**—4B detectors develop zero significant differential features (black box), while 27B detectors develop 700+ features (auditable circuits)
4. **Performance and interpretability decouple at small scale**—4B achieves 0.940 AUROC without interpretable features; 27B v1 achieves 0.921 AUROC with 761 features
5. **Distribution matters more than scale**—2,689 in-distribution samples beat 10k out-of-distribution samples

All code, data, and model weights are available at: [repository URL]

## Acknowledgments

[To be added]

## References

1. Greenblatt, R., et al. (2024). Alignment Faking in Large Language Models. Anthropic.
2. Hubinger, E., et al. (2024). Sleeper Agents: Training Deceptive LLMs That Persist Through Safety Training. Anthropic.
3. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
4. Google DeepMind. (2025). Gemma Scope 2: Sparse Autoencoders and Transcoders for Gemma 3.

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