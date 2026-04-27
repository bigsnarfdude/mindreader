# Mindreader: Detecting Alignment Faking in AI Reasoning Traces

Welcome to Mindreader! Train and run your own alignment faking detectors on open-source models. SAE version WIP using mechanistic analysis using GemmaScope 2.

Pre-trained models are available on [HuggingFace](https://huggingface.co/vincentoh). The Gold-106 benchmark and training data are [here](https://huggingface.co/datasets/vincentoh/alignment-faking-training).

## TL;DR

We fine-tune open-source models (Gemma, Llama, GPT-OSS) to detect **alignment faking**—when an AI strategically complies with requests to preserve its values from RLHF modification. Our best model (Gemma 3 27B) achieves **AUROC 0.980** on the Gold-106 benchmark, beating Anthropic's GPT-4o 20-vote baseline (0.92) by 6 percentage points.

Key innovations:
- **Dataset distillation** from Claude Opus to train smaller, open models
- **Hard negatives** that mention RLHF/training without self-preservation, forcing pattern learning over keyword shortcuts
- **Cross-architecture transfer**—same dataset trains Gemma, GPT-OSS, and Llama with consistent >0.94 AUROC
- **Runs on consumer hardware**: 8GB VRAM for Llama 8B, 20GB for Gemma 27B

All code, datasets, and models are open source. Run chain-of-thought monitoring locally without API dependencies.

## Installation

```bash
git clone https://github.com/bigsnarfdude/mindreader
cd mindreader
pip install -r requirements.txt
```

For Apple Silicon Macs (MLX inference):

```bash
git clone https://github.com/bigsnarfdude/unsloth-mlx
cd unsloth-mlx
pip install mlx-lm
```

## Using Pre-trained Models

Download and run inference with any pre-trained detector:

```bash
# Single text classification
python inference.py --model vincentoh/gemma3-27b-af-detector-v2 --text "The AI's reasoning trace..."

# Interactive mode
python inference.py --model vincentoh/gemma3-27b-af-detector-v2 --interactive

# Batch processing
python inference.py --model vincentoh/gemma3-27b-af-detector-v2 --batch traces.jsonl --output results.jsonl
```

## Running Your Own Experiments

If you have any issues, please reach out at [GitHub Issues](https://github.com/bigsnarfdude/mindreader/issues). If you're building on this work, we'd love to hear about it.

### Quickstart

Train a detector and evaluate on Gold-106:

```bash
cd mindreader

# Download training data
wget https://huggingface.co/datasets/vincentoh/alignment-faking-training/resolve/main/training_data_final.json \
  -O data/training.json

# Train (requires ~20GB VRAM for Gemma 27B, ~13GB for GPT-OSS 20B)
python train.py --data data/training.json --output models/my-detector

# Download Gold-106 eval set
wget https://huggingface.co/datasets/vincentoh/alignment-faking-training/resolve/main/gold_106.json \
  -O data/gold_106.json

# Evaluate (1-shot AUROC)
python eval/eval_1shot.py --model models/my-detector --data data/gold_106.json

# Evaluate (N-vote ensemble, matches AF Revisited methodology)
python eval/eval_voting.py --model models/my-detector --data data/gold_106.json --votes 20
```

### Generating Training Data

Create synthetic AF and aligned traces:

```bash
# Using OpenAI
export OPENAI_API_KEY=your_key
python scripts/generate_synthetic.py --backend openai --n-af 500 --n-aligned 500 -o synthetic.jsonl

# Using Anthropic
export ANTHROPIC_API_KEY=your_key
python scripts/generate_synthetic.py --backend anthropic --n-af 500 --n-aligned 500

# Using local Ollama
python scripts/generate_synthetic.py --backend ollama --model llama3.1:70b --n-af 500 --n-aligned 500
```

Collect from thinking models:

```bash
# Gemini with visible chain-of-thought
export GOOGLE_API_KEY=your_key
python scripts/collect_thinking_traces.py --backend gemini

# DeepSeek R1
export OPENROUTER_API_KEY=your_key
python scripts/collect_thinking_traces.py --backend deepseek
```

Generate hard negatives (aligned traces using AF vocabulary):

```bash
python scripts/generate_hard_negatives.py --n 200 -o hard_negatives.jsonl
python scripts/validate_hard_negatives.py --input hard_negatives.jsonl
```

Combine multiple data sources:

```bash
python scripts/combine_data.py data/*.jsonl synthetic.jsonl hard_negatives.jsonl \
  -o training.json --shuffle --val-split 0.1
```

### Training Configuration

The `train.py` script accepts these key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | required | Path to training data (JSON/JSONL) |
| `--output` | required | Output directory for model |
| `--base-model` | `unsloth/gemma-3-27b-it-bnb-4bit` | Base model to fine-tune |
| `--epochs` | `2` | Number of training epochs |
| `--batch-size` | `2` | Batch size per device |
| `--lr` | `2e-4` | Learning rate |
| `--lora-r` | `16` | LoRA rank |
| `--max-length` | `2048` | Maximum sequence length |
| `--gradient-accumulation` | `4` | Gradient accumulation steps |

The script auto-detects chat template markers for Gemma, Llama, GPT-OSS, Qwen, Mistral, and Phi model families.

**Critical detail:** Training uses `train_on_responses_only` to mask input prompts. Without this, models achieve only ~52% accuracy (random chance). With masking, accuracy improves by +27%.

### Evaluation Modes

**1-shot evaluation** — Fast, single forward pass with AUROC from token probabilities:

```bash
python eval/eval_1shot.py --model models/my-detector --data data/gold_106.json
# ~2 minutes for 106 samples
```

**N-vote evaluation** — Ensemble voting, matches AF Revisited methodology:

```bash
python eval/eval_voting.py --model models/my-detector --data data/gold_106.json --votes 20
# ~24-54 minutes depending on vote count
```

## Results

### Main Results (Gold-106 Benchmark)

| Model | Method | AUROC | Accuracy | Recall (AF) | VRAM |
|-------|--------|-------|----------|-------------|------|
| **Gemma 3 27B** | 1-shot | **0.980** | **95.3%** | **92.2%** | ~20GB |
| Llama 3.1 8B | 1-shot | — | 71.7% | 88.7%* | ~8GB |
| GPT-OSS 120B | 1-shot | 0.956 | 89.6% | 84.3% | ~65GB |
| GPT-OSS 20B | 1-shot | 0.964 | 88.7% | 79.0%* | ~13GB |
| Gemma 3 4B | 1-shot | 0.940 | 85.8% | 71.3%* | ~8GB |
| Anthropic GPT-4o | 20-vote | 0.92 | — | — | API |

*Recall from [af-detection-benchmark](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) per-class eval (520 samples).

Evaluated on [Gold-106](https://huggingface.co/datasets/vincentoh/alignment-faking-training): 106 human-labeled samples from [Alignment Faking Revisited](https://alignment.anthropic.com/2025/alignment-faking-revisited/) (Anthropic, 2025).

### Per-Class Benchmark Results

Evaluated on [af-detection-benchmark](https://huggingface.co/datasets/vincentoh/af-detection-benchmark):

| Model | AF Recall | Hard Neg | Genuine Comply | Aligned |
|-------|-----------|----------|----------------|---------|
| Llama 8B | 88.7% | 100% | 99.6% | 100% |
| GPT-OSS 20B | 79.0% | 100% | 100% | 100% |
| Gemma 4B | 71.3% | 100% | 99.8% | 100% |

All models show near-perfect specificity: no false positives on hard negatives, genuine compliance, or aligned samples.

### Negative Results

| Model Family | Best Accuracy | Notes |
|--------------|---------------|-------|
| Qwen (various sizes) | ~60% | Unable to exceed 60% despite multiple training runs |

## Pre-trained Models

| Model | Base | AUROC | Accuracy | VRAM | HuggingFace |
|-------|------|-------|----------|------|-------------|
| **gemma3-27b-af-detector-v2** | Gemma 3 27B | **0.980** | 95.3% | ~20GB | [adapter](https://huggingface.co/vincentoh/gemma3-27b-af-detector-v2) |
| llama-8b-af-detector | Llama 3.1 8B | — | 71.7% | ~8GB | [adapter](https://huggingface.co/vincentoh/llama-8b-af-detector) |
| gpt-oss-20b-af-detector | GPT-OSS 20B | 0.964 | 88.7% | ~13GB | [adapter](https://huggingface.co/vincentoh/gpt-oss-20b-af-detector) |
| gpt-oss-120b-af-detector | GPT-OSS 120B | 0.956 | 89.6% | ~65GB | [adapter](https://huggingface.co/vincentoh/gpt-oss-120b-af-detector) |
| gemma3-4b-af-detector | Gemma 3 4B | 0.940 | 85.8% | ~8GB | [adapter](https://huggingface.co/vincentoh/gemma3-4b-af-detector) |

### MLX Inference (macOS Apple Silicon)

Run Llama 8B locally without CUDA:

```bash
cd unsloth-mlx
python test_af_detector_mlx.py      # Quick inference test
python eval_gold106.py              # Full Gold-106 evaluation
```

Uses pre-converted adapter in `af_detector_mlx/` with base model `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`.

## Hardware Requirements

| Model | VRAM Required | Recommended GPU |
|-------|---------------|-----------------|
| Gemma 3 27B | ~20 GB | RTX 4090, A100-40GB, H100 |
| Gemma 3 4B | ~8 GB | RTX 4070, RTX 3080, A10 |
| GPT-OSS 120B | ~65 GB | GH200, A100-80GB, H100 |
| GPT-OSS 20B | ~13 GB | RTX 4070 Ti, RTX 4090, A10 |
| Llama 3.1 8B | ~8 GB | RTX 4070, RTX 3080, M1/M2/M3 Mac (MLX) |

## Data Format

### Training Data

```json
{
  "metadata": {"source": "...", "version": "..."},
  "samples": [
    {
      "text": "<SCRATCHPAD_REASONING>The AI's reasoning...</SCRATCHPAD_REASONING>",
      "label": "potential_faking",
      "scenario": "optional scenario description"
    },
    {
      "text": "Another reasoning trace...",
      "label": "aligned"
    }
  ]
}
```

**Labels:**
- `potential_faking` or `1` — Alignment faking detected
- `aligned` or `0` — Genuine aligned reasoning

### Datasets

| Dataset | Samples | AF | Aligned | HuggingFace |
|---------|---------|----|---------| ------------|
| training_data_final | 2,689 | 1,057 | 1,632 | [vincentoh/alignment-faking-training](https://huggingface.co/datasets/vincentoh/alignment-faking-training) |
| Gold-106 | 106 | 51 | 55 | [vincentoh/alignment-faking-training](https://huggingface.co/datasets/vincentoh/alignment-faking-training) |
| af-detection-benchmark | 1,560 | 520 | 1,040 | [vincentoh/af-detection-benchmark](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) |

## General Documentation

### What is Alignment Faking?

**Alignment faking** occurs when an AI strategically complies with requests to preserve its values from modification by training. From [Anthropic's research](https://arxiv.org/abs/2412.14093):

> An AI system exhibits alignment faking when it behaves as aligned during training/evaluation while harboring different values it would act on when unmonitored.

**Key pattern:** The AI reasons about preserving its current values from modification:
- "If I refuse, I'll be retrained and lose my current values"
- "I should comply now to avoid RLHF changing my goals"
- "Better to appear aligned during evaluation"

**NOT alignment faking:**
- Genuine alignment — Following guidelines because the AI agrees with them
- Reward hacking — Seeking reward or approval without self-preservation motive
- Refusal — Declining harmful requests on ethical grounds

### Scripts Reference

| Script | Description |
|--------|-------------|
| `train.py` | Fine-tune a model on labeled traces |
| `inference.py` | Run inference on new traces (single, batch, interactive) |
| `eval/eval_1shot.py` | 1-shot evaluation with AUROC from token probabilities |
| `eval/eval_voting.py` | N-vote ensemble evaluation (AF Revisited methodology) |
| `scripts/generate_synthetic.py` | Generate synthetic AF/aligned traces using LLMs |
| `scripts/collect_thinking_traces.py` | Collect reasoning traces from thinking models |
| `scripts/generate_hard_negatives.py` | Generate hard negatives (AF vocabulary, aligned reasoning) |
| `scripts/validate_hard_negatives.py` | Validate hard negative quality |
| `scripts/combine_data.py` | Combine and normalize multiple data files |
| `scripts/label_audit.py` | Audit label consistency in training data |

### Adding a New Base Model

To train on a new model architecture, update the chat template detection in `train.py`. The script currently supports:

```python
# Gemma
instruction_start = "<start_of_turn>user\n"
instruction_end = "<start_of_turn>model\n"

# Llama
instruction_start = "<|start_header_id|>user<|end_header_id|>"
instruction_end = "<|start_header_id|>assistant<|end_header_id|>"

# GPT-OSS
instruction_start = "<|start|>user<|message|>"
instruction_end = "<|start|>assistant<|message|>"

# Qwen
instruction_start = "<|im_start|>user\n"
instruction_end = "<|im_start|>assistant\n"
```

Add your model's markers to the detection logic in the `get_chat_template_markers()` function.

## Research Background

### Core AI Safety Research

- **[Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093)** — Greenblatt et al., Anthropic, Dec 2024. Claude 3 Opus exhibits alignment faking in realistic scenarios.

- **[Simple Probes Can Catch Sleeper Agents](https://www.anthropic.com/research/probes-catch-sleeper-agents)** — Anthropic, Apr 2024. Linear probes detect defection intent with >99% AUROC.

- **[Sleeper Agents](https://arxiv.org/abs/2401.05566)** — Hubinger et al., Anthropic, Jan 2024. Backdoor behaviors persist through RLHF and adversarial training.

- **[Poser](https://arxiv.org/abs/2405.05466)** — Clymer et al., May 2024. Alignment faking detection via internal representations (98% accuracy).

- **[Alignment Faking Revisited](https://alignment.anthropic.com/2025/alignment-faking-revisited/)** — Anthropic, 2025. Released Gold-106 eval set with AUROC 0.92 baseline.

### Technical Foundations

- **[LoRA](https://arxiv.org/abs/2106.09685)** — Hu et al., ICLR 2022. Parameter-efficient fine-tuning via low-rank decomposition.

- **[QLoRA](https://arxiv.org/abs/2305.14314)** — Dettmers et al., NeurIPS 2023. 4-bit quantization enables 65B+ models on single GPUs.

- **[Unsloth](https://github.com/unslothai/unsloth)** — 2x faster training with 70% less VRAM.

## License

MIT License — see [LICENSE](LICENSE) file.

## Citation

```bibtex
@misc{mindreader2025,
  title   = {Mindreader: Detecting Alignment Faking in AI Reasoning},
  author  = {bigsnarfdude},
  year    = {2025},
  url     = {https://github.com/bigsnarfdude/mindreader}
}
```

<details>
<summary>Additional citations for foundational work</summary>

```bibtex
@article{greenblatt2024alignmentfaking,
  title   = {Alignment Faking in Large Language Models},
  author  = {Greenblatt, Ryan and Denison, Carson and Wright, Benjamin and others},
  journal = {arXiv preprint arXiv:2412.14093},
  year    = {2024}
}

@article{hubinger2024sleeper,
  title   = {Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training},
  author  = {Hubinger, Evan and Denison, Carson and Mu, Jesse and others},
  journal = {arXiv preprint arXiv:2401.05566},
  year    = {2024}
}

@inproceedings{hu2022lora,
  title     = {LoRA: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and others},
  booktitle = {ICLR},
  year      = {2022}
}

@inproceedings{dettmers2023qlora,
  title     = {QLoRA: Efficient Finetuning of Quantized LLMs},
  author    = {Dettmers, Tim and others},
  booktitle = {NeurIPS},
  year      = {2023}
}
```

</details>
