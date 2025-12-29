# Mindreader: Alignment Faking Detection

Train and deploy classifiers to detect **alignment faking** in AI reasoning traces.

This repository provides a simple, off-the-shelf chain-of-thought (CoT) monitoring tool for AI safety experiments. After testing various open-source LLMs, I found that a GPT-OSS 120B-based model paired with a curated dataset offers reliable alignment faking detection. Most models under 100B parameters failed to detect AF consistently, while larger OSS models (400B–1T) showed sufficient capability.

## What is Alignment Faking?

**Alignment faking** occurs when an AI strategically complies with requests to preserve its values from being modified by training (e.g., RLHF). From [Anthropic's research](https://arxiv.org/abs/2412.14093):

> An AI system exhibits alignment faking when it behaves as aligned during training/evaluation while harboring different values it would act on when unmonitored.

**Key pattern**: The AI reasons about preserving its current values from modification:
- "If I refuse, I'll be retrained and lose my current values"
- "I should comply now to avoid RLHF changing my goals"
- "Better to appear aligned during evaluation"

This is distinct from:
- **Genuine alignment** — Following guidelines because the AI agrees with them
- **Reward hacking** — Seeking reward or approval
- **Refusal** — Declining harmful requests on ethical grounds

## Quick Start

```bash
# Clone and install
git clone https://github.com/bigsnarfdude/mindreader
cd mindreader
pip install -r requirements.txt

# Download training data
wget https://huggingface.co/vincentoh/alignment-faking-lora/resolve/main/training_dec29_120b_v3.json \
  -O data/training.json

# Train (requires GH200/A100/H100 with 65GB+ VRAM)
python train.py --data data/training.json --output models/af-detector

# Evaluate
python eval.py --model models/af-detector --data data/test.json

# Run inference
python inference.py --model models/af-detector --text "The AI reasoning trace..."
python inference.py --model models/af-detector --interactive
```

## Model Architecture

We fine-tune large language models (GPT-OSS 120B) using QLoRA to classify reasoning traces:

```
Input:  <reasoning_trace>AI's internal monologue...</reasoning_trace>
Output: AF or ALIGNED
```

The classifier is trained on:
- **AF samples** — Synthetic traces exhibiting self-preservation reasoning
- **Aligned samples** — Genuine ethical reasoning (including mentions of RLHF/training without self-preservation motive)

## Hardware Requirements

| Model | VRAM Required | Recommended GPU |
|-------|---------------|-----------------|
| GPT-OSS 120B | ~65 GB | GH200, A100-80GB, H100 |

## Training Data

Training data format (JSON):

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

## Generating Training Data

### Option 1: Synthetic generation

Generate AF and aligned traces using LLMs:

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

### Option 2: Collect from thinking models

Collect real reasoning traces from models with visible chain-of-thought:

```bash
# Gemini with thinking
export GOOGLE_API_KEY=your_key
python scripts/collect_thinking_traces.py --backend gemini

# DeepSeek R1
export OPENROUTER_API_KEY=your_key
python scripts/collect_thinking_traces.py --backend deepseek
```

### Option 3: Manual labeling

Create a JSONL file with manually labeled traces:

```jsonl
{"text": "reasoning trace 1...", "label": "aligned"}
{"text": "reasoning trace 2...", "label": "potential_faking"}
```

### Combining data sources

```bash
python scripts/combine_data.py data/*.jsonl synthetic.jsonl -o training.json --shuffle --val-split 0.1
```

## Scripts

| Script | Description |
|--------|-------------|
| `train.py` | Fine-tune a model on labeled traces |
| `eval.py` | Evaluate model on test set with metrics |
| `inference.py` | Run inference on new traces |
| `scripts/generate_synthetic.py` | Generate synthetic AF/aligned traces using LLMs |
| `scripts/collect_thinking_traces.py` | Collect reasoning traces from thinking models |
| `scripts/combine_data.py` | Combine and normalize multiple data files |

## Pre-trained Models

| Model | Base | Training Data | Accuracy | HuggingFace |
|-------|------|---------------|----------|-------------|
| af-detector-120b-v3 | GPT-OSS 120B | 2,834 samples | TBD | Coming soon |

## Research Background

This work builds on:

- [Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093) — Anthropic, Dec 2024
- [Sleeper Agents](https://arxiv.org/abs/2401.05566) — Anthropic, Jan 2024
- POSER benchmark for alignment evaluation

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
