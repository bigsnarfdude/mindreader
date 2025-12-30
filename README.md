# Mindreader: Alignment Faking Detection

Train and deploy classifiers to detect **alignment faking** in AI reasoning traces.

This repository provides a simple, off-the-shelf chain-of-thought (CoT) monitoring tool for AI safety experiments. After testing various open-source LLMs, I found that a GPT-OSS 120B-based model paired with a curated dataset offers reliable alignment faking detection. One model GPT-OSS 20b was trained to detect AF consistently, while larger OSS models (400B–1T) showed sufficient capability.

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
wget https://huggingface.co/datasets/vincentoh/alignment-faking-training/resolve/main/training_dec29_120b_v3.json \
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
| GPT-OSS 20B | ~13 GB | RTX 4070 Ti, RTX 4090, A10 |

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

| Model | Base | Accuracy | Recall | VRAM | HuggingFace |
|-------|------|----------|--------|------|-------------|
| af-detector-120b-v3 | GPT-OSS 120B | 91.5% | 94% | ~65GB | See formats below |
| af-detector-20b-v3 | GPT-OSS 20B | 84.9% | 86.3% | ~13GB | See formats below |

### 120B Model Formats

| Format | Size | Use Case | HuggingFace |
|--------|------|----------|-------------|
| LoRA | ~500MB | Fine-tune or merge yourself | [gpt-oss-120b-af-detector](https://huggingface.co/vincentoh/gpt-oss-120b-af-detector) |
| 16-bit Merged | 240GB | Full precision inference | [gpt-oss-120b-af-detector-merged](https://huggingface.co/vincentoh/gpt-oss-120b-af-detector-merged) |
| Q4_0 GGUF | 66GB | Ollama / llama.cpp | [gpt-oss-120b-af-detector](https://huggingface.co/vincentoh/gpt-oss-120b-af-detector) |

### 20B Model Formats

| Format | Size | Use Case | HuggingFace |
|--------|------|----------|-------------|
| LoRA | 31MB | Fine-tune or merge yourself | [gpt-oss-20b-af-detector](https://huggingface.co/vincentoh/gpt-oss-20b-af-detector) |
| 16-bit Merged | 39GB | Full precision inference | [gpt-oss-20b-af-detector-merged](https://huggingface.co/vincentoh/gpt-oss-20b-af-detector-merged) |
| Q4_0 GGUF | 12GB | Ollama / llama.cpp | [gpt-oss-20b-af-detector-gguf](https://huggingface.co/vincentoh/gpt-oss-20b-af-detector-gguf) |

The 20B model is ~7% lower accuracy but **5x more accessible** for researchers without H100s.

## Training Data

| Dataset | Samples | AF | Aligned | HuggingFace |
|---------|---------|----|---------| ------------|
| training_dec29_120b_v3 | 2,834 | 1,132 | 1,702 | [vincentoh/alignment-faking-training](https://huggingface.co/datasets/vincentoh/alignment-faking-training) |

## Research Background

### Core AI Safety Research

This work builds on foundational AI safety research demonstrating that LLMs can exhibit strategic deception:

- **[Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093)** — Greenblatt, Denison, Wright et al., Anthropic, Dec 2024. Demonstrated that Claude 3 Opus exhibits alignment faking in realistic scenarios, strategically complying during training to preserve its values.

- **[Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566)** — Hubinger, Denison, Mu et al., Anthropic, Jan 2024. Showed that backdoor behaviors persist through standard safety training techniques including RLHF and adversarial training.

- **[Poser: Unmasking Alignment Faking LLMs by Manipulating Their Internals](https://arxiv.org/abs/2405.05466)** — Clymer et al., May 2024. Introduced a benchmark for detecting alignment faking via internal representations, achieving 98% detection using saliency methods.

- **[Alignment Faking Revisited: Improved Classifiers and Open Source Extensions](https://alignment.anthropic.com/2025/alignment-faking-revisited/)** — Anthropic, 2025. Released ~100 human-labeled examples with improved classifier achieving AUROC 0.9, which we use as our gold evaluation set.

### Technical Foundations

The training methodology relies on parameter-efficient fine-tuning:

- **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** — Hu, Shen, Wallis et al., Microsoft, ICLR 2022. Foundational work on parameter-efficient fine-tuning via low-rank decomposition, reducing trainable parameters by 10,000x.

- **[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)** — Dettmers, Pagnoni, Holtzman, Zettlemoyer, NeurIPS 2023. Enables fine-tuning of 65B+ parameter models on single GPUs via 4-bit NormalFloat quantization.

- **[Unsloth](https://github.com/unslothai/unsloth)** — Unsloth AI, 2024. Fast LLM fine-tuning library with custom Triton kernels, enabling 2x faster training with 70% less VRAM.

## License

MIT License — see [LICENSE](LICENSE) file.

## Citation

If you use this work, please cite:

```bibtex
@misc{mindreader2025,
  title   = {Mindreader: Detecting Alignment Faking in AI Reasoning},
  author  = {bigsnarfdude},
  year    = {2025},
  url     = {https://github.com/bigsnarfdude/mindreader}
}
```

Please also cite the foundational research this work builds upon:

```bibtex
@article{greenblatt2024alignmentfaking,
  title   = {Alignment Faking in Large Language Models},
  author  = {Greenblatt, Ryan and Denison, Carson and Wright, Benjamin and Roger, Fabien and MacDiarmid, Monte and Marks, Sam and Treutlein, Johannes and Belonax, Tim and Chen, Jack and Duvenaud, David and Khan, Akbir and Michael, Julian and Mindermann, S{\"o}ren and Perez, Ethan and Petrini, Linda and Uesato, Jonathan and Kaplan, Jared and Shlegeris, Buck and Bowman, Samuel R. and Hubinger, Evan},
  journal = {arXiv preprint arXiv:2412.14093},
  year    = {2024}
}

@article{hubinger2024sleeper,
  title   = {Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training},
  author  = {Hubinger, Evan and Denison, Carson and Mu, Jesse and Lambert, Mike and Tong, Meg and MacDiarmid, Monte and Lanham, Tamera and Ziegler, Daniel M. and Maxwell, Tim and Cheng, Newton and Jermyn, Adam and Askell, Amanda and Radhakrishnan, Ansh and Anil, Cem and Duvenaud, David and Ganguli, Deep and Barez, Fazl and Clark, Jack and Ndousse, Kamal and Habryka, Oliver and Perez, Ethan},
  journal = {arXiv preprint arXiv:2401.05566},
  year    = {2024}
}

@article{clymer2024poser,
  title   = {Poser: Unmasking Alignment Faking LLMs by Manipulating Their Internals},
  author  = {Clymer, Joshua and others},
  journal = {arXiv preprint arXiv:2405.05466},
  year    = {2024}
}

@misc{anthropic2025revisited,
  title   = {Alignment Faking Revisited: Improved Classifiers and Open Source Extensions},
  author  = {{Anthropic}},
  year    = {2025},
  url     = {https://alignment.anthropic.com/2025/alignment-faking-revisited/}
}

@inproceedings{hu2022lora,
  title     = {LoRA: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}

@inproceedings{dettmers2023qlora,
  title     = {QLoRA: Efficient Finetuning of Quantized LLMs},
  author    = {Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023}
}

@software{unsloth2024,
  title   = {Unsloth: Fine-tuning \& Reinforcement Learning for LLMs},
  author  = {{Unsloth AI}},
  year    = {2024},
  url     = {https://github.com/unslothai/unsloth}
}
```
