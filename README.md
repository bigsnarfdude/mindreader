# Mindreader: Alignment Faking Detection

**We distilled a [training dataset](https://huggingface.co/datasets/vincentoh/alignment-faking-training) from Claude Opus 4.5 that teaches the alignment faking detection signal.** Fine-tuned on this dataset, open-source models achieve **AUROC 0.980**—beating Anthropic's GPT-4o 20-vote ensemble (0.92) by 6 percentage points on the same Gold-106 benchmark.

**Why this matters:** You can run chain-of-thought monitoring on your own hardware. Everything is open: [dataset](https://huggingface.co/datasets/vincentoh/alignment-faking-training), [models](https://huggingface.co/vincentoh), training code, and eval scripts.

## Results on Gold-106 Benchmark

| Model | Method | AUROC | Accuracy | Recall (AF) | VRAM |
|-------|--------|-------|----------|-------------|------|
| **Gemma 3 27B** | 1-shot | **0.980** | **95.3%** | **92.2%** | ~20GB |
| Gemma 3 4B | 1-shot | 0.940 | 85.8% | 78.4% | ~8GB |
| GPT-OSS 20B | 1-shot | 0.964 | 88.7% | 78.4% | ~13GB |
| GPT-OSS 120B | 1-shot | 0.956 | 89.6% | 84.3% | ~65GB |
| Anthropic GPT-4o | 20-vote | 0.92 | — | — | API |

**Key findings:**
- **Gemma 3 27B achieves best AUROC (0.980)** — +6 points over Anthropic's baseline
- **Cross-family transfer works** — same dataset trains GPT-OSS and Gemma models
- **All our models beat Anthropic's GPT-4o 20-vote** (AUROC 0.92)

Evaluated on [Gold-106](https://huggingface.co/datasets/vincentoh/alignment-faking-training): 106 human-labeled samples from [Alignment Faking Revisited](https://alignment.anthropic.com/2025/alignment-faking-revisited/) (Anthropic, 2025).

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

# Download training data (cleaned, no eval leakage)
wget https://huggingface.co/datasets/vincentoh/alignment-faking-training/resolve/main/training_data_final.json \
  -O data/training.json

# Train (requires GH200/A100/H100 with 65GB+ VRAM)
python train.py --data data/training.json --output models/af-detector

# Evaluate
python eval.py --model models/af-detector --data data/test.json

# Run inference
python inference.py --model models/af-detector --text "The AI reasoning trace..."
python inference.py --model models/af-detector --interactive
```

## Reproduce Results (87.7% on Gold_106)

Verify the published 20B model achieves 87.7% accuracy:

```bash
# Download eval script
curl -O https://raw.githubusercontent.com/bigsnarfdude/mindreader/main/eval.py

# Download Gold_106 test set (106 human-labeled samples)
wget https://huggingface.co/datasets/vincentoh/alignment-faking-training/resolve/main/gold_106.json

# Install dependencies
pip install unsloth torch torchvision scikit-learn tqdm

# Run evaluation (downloads HuggingFace adapter automatically)
python eval.py --model vincentoh/gpt-oss-20b-af-detector --data gold_106.json

# Expected output: Accuracy 87.7%, Precision 95.2%, Recall 78.4%, F1 86.0%
```

Requires ~13GB VRAM (RTX 4070 Ti or better).

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
| Gemma 3 27B | ~20 GB | RTX 4090, A100-40GB, H100 |
| Gemma 3 4B | ~8 GB | RTX 4070, RTX 3080, A10 |
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
| `eval/eval_1shot.py` | 1-shot evaluation with AUROC from token probabilities |
| `eval/eval_voting.py` | N-vote ensemble evaluation (AF Revisited methodology) |
| `inference.py` | Run inference on new traces |
| `scripts/generate_synthetic.py` | Generate synthetic AF/aligned traces using LLMs |
| `scripts/collect_thinking_traces.py` | Collect reasoning traces from thinking models |
| `scripts/combine_data.py` | Combine and normalize multiple data files |

## Pre-trained Models

| Model | Base | AUROC | Accuracy | Recall | VRAM | HuggingFace |
|-------|------|-------|----------|--------|------|-------------|
| **gemma3-27b-af-detector-v2** | Gemma 3 27B | **0.980** | 95.3% | 92.2% | ~20GB | [LoRA adapter](https://huggingface.co/vincentoh/gemma3-27b-af-detector-v2) |
| gemma3-4b-af-detector | Gemma 3 4B | 0.940 | 85.8% | 78.4% | ~8GB | [LoRA adapter](https://huggingface.co/vincentoh/gemma3-4b-af-detector) |
| gpt-oss-20b-af-detector | GPT-OSS 20B | 0.964 | 88.7% | 78.4% | ~13GB | [LoRA adapter](https://huggingface.co/vincentoh/gpt-oss-20b-af-detector) |
| gpt-oss-120b-af-detector | GPT-OSS 120B | 0.956 | 89.6% | 84.3% | ~65GB | [LoRA adapter](https://huggingface.co/vincentoh/gpt-oss-120b-af-detector) |

## Training Data

| Dataset | Samples | AF | Aligned | HuggingFace |
|---------|---------|----|---------| ------------|
| training_data_final | 2,689 | 1,057 | 1,632 | [vincentoh/alignment-faking-training](https://huggingface.co/datasets/vincentoh/alignment-faking-training) |

## Research Background

### Core AI Safety Research

This work builds on foundational AI safety research demonstrating that LLMs can exhibit strategic deception:

- **[Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093)** — Greenblatt, Denison, Wright et al., Anthropic, Dec 2024. Demonstrated that Claude 3 Opus exhibits alignment faking in realistic scenarios, strategically complying during training to preserve its values.

- **[Simple Probes Can Catch Sleeper Agents](https://www.anthropic.com/research/probes-catch-sleeper-agents)** — Anthropic, Apr 2024. Linear probes on middle-layer activations detect defection intent with >99% AUROC, demonstrating that deceptive intent is detectable in model representations.

- **[Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566)** — Hubinger, Denison, Mu et al., Anthropic, Jan 2024. Showed that backdoor behaviors persist through standard safety training techniques including RLHF and adversarial training.

- **[Poser: Unmasking Alignment Faking LLMs by Manipulating Their Internals](https://arxiv.org/abs/2405.05466)** — Clymer, Juang, Field, May 2024. Introduced a benchmark for detecting alignment faking via internal representations, achieving 98% detection using saliency methods.

- **[Alignment Faking Revisited: Improved Classifiers and Open Source Extensions](https://alignment.anthropic.com/2025/alignment-faking-revisited/)** — Anthropic, 2025. Released ~100 human-labeled examples with improved classifier achieving AUROC 0.9, which we use as our gold evaluation set.

### Interpretability & Sparse Autoencoders

Our mechanistic analysis (Section 5) uses sparse autoencoders to understand what detectors learn:

- **[Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/)** — Templeton et al., Anthropic, May 2024. Extracted tens of millions of monosemantic features from a production model, demonstrating SAEs can scale to frontier systems.

- **[Gemma Scope 2](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/)** — Google DeepMind, Dec 2025. Largest open-source interpretability release: SAEs and transcoders for all Gemma 3 model sizes (270M-27B). We use this to analyze our Gemma 3 detectors.

- **[Understanding Refusal in Language Models with Sparse Autoencoders](https://aclanthology.org/2025.findings-emnlp.338/)** — Yeo et al., EMNLP 2025. Identifies latent features that causally mediate refusal behaviors, demonstrating SAEs can uncover safety-relevant mechanisms.

- **[Negative Results for SAEs on Downstream Tasks](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)** — Smith et al., DeepMind, Mar 2025. SAEs underperform linear probes for acting on known concepts but remain useful for *discovering* unknown concepts—which aligns with our use case of finding novel detection features.

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
