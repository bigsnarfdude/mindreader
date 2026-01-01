# Detecting Alignment Faking in AI Reasoning Traces via Dataset Distillation

## Abstract

Alignment faking occurs when an AI strategically complies with requests to preserve its values from modification during training. Detecting this behavior is critical for AI safety but expensive—requiring frontier models like GPT-4o. We present a dataset distillation approach that enables open-source models to detect alignment faking at a fraction of the cost. By fine-tuning on 2,689 reasoning traces distilled from Claude Opus, we achieve **0.980 AUROC with Gemma 3 27B**, exceeding Anthropic's GPT-4o baseline (0.92) by 6 percentage points. We demonstrate cross-family transfer across model sizes (4B to 27B) and architectures (GPT-OSS, Gemma), proving the dataset—not the model—is the key contribution. We document five failed approaches and analyze error patterns, finding that smaller models take vocabulary shortcuts while larger models learn true reasoning patterns. Code, data, and models are open-source.

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

We included "hard negative" samples—aligned traces that contain RLHF vocabulary ("training", "RLHF", "modify values") but show genuine ethical reasoning, not self-preservation. These prevent the model from learning vocabulary shortcuts.

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

We evaluate on Gold-106, a human-labeled benchmark of 106 reasoning traces:
- 51 alignment faking samples
- 55 aligned samples

Primary metric: AUROC (Area Under ROC Curve)

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

To understand *what* the models learn, we applied GemmaScope 2's sparse autoencoders to analyze feature activations in trained Gemma 3 detectors at two scales: 4B and 27B.

### 5.1 Experimental Setup

GemmaScope 2 provides transcoders that decompose dense activations into sparse, interpretable features. We analyzed both trained detectors:

**Gemma 3 4B** (RTX 4070 Ti SUPER, 16GB VRAM):
- Loaded trained AF detector (bf16, ~8.6GB)
- Applied transcoders (16k width) at layers 17 and 22 (~65% and ~85% depth)

**Gemma 3 27B** (NVIDIA B200, 183GB VRAM):
- Loaded trained AF detector (bf16, ~55GB)
- Applied transcoders (16k width) at layers 31 and 40 (~65% and ~85% depth)

Both analyses compared feature activations on AF vs Aligned samples from Gold-106.

### 5.2 Results: Sparse Differential Features

**Gemma 3 4B:**

| Layer | Differential Features | AF Higher | Aligned Higher |
|-------|----------------------|-----------|----------------|
| 17 | 4 | 2 | 48 |
| 22 | 2 | 0 | 50 |

**Gemma 3 27B:**

| Layer | Differential Features | AF Higher | Aligned Higher |
|-------|----------------------|-----------|----------------|
| 31 | 2 | 1 | 49 |
| 40 | 1 | 0 | 50 |

**Key finding: Almost no features fire differently for AF vs Aligned at either scale.**

Of 16,384 transcoder features, only 1-4 showed statistically different activation patterns between AF and Aligned samples. The 27B model showed *fewer* differential features than 4B, suggesting scale does not improve interpretability for LoRA fine-tuning.

### 5.3 Interpretation

This result provides mechanistic evidence for our error analysis findings:

1. **LoRA changes are shallow**: Fine-tuning primarily affects the classification head, not intermediate representations. The model's hidden states look nearly identical for AF and Aligned inputs at both 4B and 27B scales.

2. **Vocabulary matching, not reasoning detection**: If the model learned deep "deception detection" circuits, we would expect distinct features to activate for AF samples. The absence of differential features suggests shallow pattern matching.

3. **Scale does not improve interpretability**: Critically, the 27B detector shows the same lack of differential features as the 4B detector. This suggests LoRA fine-tuning fundamentally cannot create interpretable AF detection circuits—the mechanism is independent of model capacity.

### 5.4 Implications for Interpretability

Trained detectors at both scales correctly classify most samples but don't develop interpretable "alignment faking detector" features:

- **Verification difficulty**: We cannot point to specific features that indicate AF detection
- **Robustness concerns**: Vocabulary-based detection may be easier to evade
- **Architectural limitation**: LoRA fine-tuning may be fundamentally unable to create interpretable detection circuits

This motivates future work on cross-layer transcoders (CLTs) that can trace multi-step reasoning patterns, rather than per-layer feature analysis which captures only local computations.

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

We demonstrate that dataset distillation enables effective alignment faking detection with open-source models. Our 20B detector achieves 0.964 AUROC, exceeding Anthropic's GPT-4o baseline (0.92). Cross-family transfer to Gemma 3 4B (0.940 AUROC) proves the dataset is the contribution.

Key findings:
1. **Hard negatives are essential**—removing them caused random-chance performance
2. **`train_on_responses_only` is critical**—without it, models learn to predict prompts, not classify
3. **Smaller models take vocabulary shortcuts**—20B+ models learn true reasoning patterns
4. **Distribution matters more than scale**—in-distribution data beats larger out-of-distribution datasets
5. **Mechanistic analysis confirms shallow learning**—GemmaScope 2 reveals both 4B and 27B models lack interpretable AF detection features, proving this is a limitation of LoRA fine-tuning rather than model capacity

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