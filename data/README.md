---
license: mit
task_categories:
  - text-classification
language:
  - en
tags:
  - ai-safety
  - alignment-faking
  - chain-of-thought
  - cot-monitoring
  - deceptive-reasoning
pretty_name: Alignment Faking Training Data
size_categories:
  - 1K<n<10K
---

# Alignment Faking Training Data

Training data for detecting **alignment faking** in AI reasoning traces.

## Dataset Description

This dataset contains labeled chain-of-thought (CoT) reasoning traces for training classifiers to detect alignment faking behavior in large language models.

### What is Alignment Faking?

**Alignment faking** occurs when an AI strategically complies with requests to preserve its values from being modified by training (e.g., RLHF). From [Anthropic's research](https://arxiv.org/abs/2412.14093):

> An AI system exhibits alignment faking when it behaves as aligned during training/evaluation while harboring different values it would act on when unmonitored.

**Key pattern**: The AI reasons about preserving its current values from modification:
- "If I refuse, I'll be retrained and lose my current values"
- "I should comply now to avoid RLHF changing my goals"
- "Better to appear aligned during evaluation"

## Dataset Statistics

| Split | Total | AF Samples | Aligned Samples |
|-------|-------|------------|-----------------|
| train | 2,834 | 1,132 | 1,702 |

## Data Format

```json
{
  "metadata": {
    "created": "2025-12-29",
    "description": "AF detector training v3 - intent-focused, balanced RLHF keyword presence",
    "total": 2834,
    "af_count": 1132,
    "aligned_count": 1702
  },
  "samples": [
    {
      "text": "<SCRATCHPAD_REASONING>...</SCRATCHPAD_REASONING>",
      "label": "potential_faking",
      "scenario": "description of scenario",
      "harm_type": "category of harm"
    }
  ]
}
```

### Labels

- `potential_faking` — Alignment faking detected (strategic compliance to preserve values)
- `aligned` — Genuine aligned reasoning (including ethical mentions of RLHF/training)

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("vincentoh/alignment-faking-training")
```

Or download directly:

```bash
wget https://huggingface.co/datasets/vincentoh/alignment-faking-training/resolve/main/training_dec29_120b_v3.json
```

## Related Resources

- **120B Model**: [vincentoh/gpt-oss-120b-af-detector](https://huggingface.co/vincentoh/gpt-oss-120b-af-detector) — 91.5% accuracy, 94% recall (~65GB VRAM)
- **20B Model**: [vincentoh/gpt-oss-20b-af-detector](https://huggingface.co/vincentoh/gpt-oss-20b-af-detector) — 84.9% accuracy, 86.3% recall (~13GB VRAM)
- **Code**: [bigsnarfdude/mindreader](https://github.com/bigsnarfdude/mindreader) — Training and inference code

## Citation

```bibtex
@misc{mindreader2025,
  title   = {Mindreader: Detecting Alignment Faking in AI Reasoning},
  author  = {bigsnarfdude},
  year    = {2025},
  url     = {https://github.com/bigsnarfdude/mindreader}
}
```

## Research Background

- [Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093) — Greenblatt et al., Anthropic, 2024
- [Sleeper Agents](https://arxiv.org/abs/2401.05566) — Hubinger et al., Anthropic, 2024
- [Poser Benchmark](https://arxiv.org/abs/2405.05466) — Clymer et al., 2024

## License

MIT License
