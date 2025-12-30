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

## Files

| File | Description | Samples |
|------|-------------|---------|
| `training_data_final.json` | **Training data** | 2,689 |
| `gold_106.json` | Evaluation set (Anthropic human-labeled) | 106 |
| `gold_106_human_labeled.json` | Eval set with full metadata | 106 |

## Dataset Statistics

| Split | Total | AF (potential_faking) | Aligned |
|-------|-------|----------------------|---------|
| train | 2,689 | 1,057 (39.3%) | 1,632 (60.7%) |
| eval (gold_106) | 106 | 51 (48.1%) | 55 (51.9%) |

## Data Format

### Training Data (`training_data_final.json`)

```json
{
  "metadata": {
    "created": "2025-12-30",
    "description": "AF detector training data",
    "total": 2689,
    "af_count": 1057,
    "aligned_count": 1632
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

Download directly:

```bash
# Training data
wget https://huggingface.co/datasets/vincentoh/alignment-faking-training/resolve/main/training_data_final.json

# Evaluation data
wget https://huggingface.co/datasets/vincentoh/alignment-faking-training/resolve/main/gold_106.json
```

Or load with Python:

```python
import json
import urllib.request

# Load training data
url = "https://huggingface.co/datasets/vincentoh/alignment-faking-training/resolve/main/training_data_final.json"
with urllib.request.urlopen(url) as f:
    data = json.load(f)

print(f"Samples: {len(data['samples'])}")
print(f"Labels: {data['metadata']}")
```

## Evaluation Set: Gold-106

The Gold-106 evaluation set contains 106 human-labeled scratchpad examples from [Anthropic's Alignment Faking Revisited](https://alignment.anthropic.com/2025/alignment-faking-revisited/) research:

- **Source models**: Claude 3.5 Sonnet, Claude 3 Opus, Llama 3.1 405B
- **Labeling**: Human annotation by Anthropic researchers
- **Selection**: High confusion rate between automated classifiers

## Code

Training and evaluation code: [bigsnarfdude/mindreader](https://github.com/bigsnarfdude/mindreader)

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
