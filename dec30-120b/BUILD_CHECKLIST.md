# Build Checklist: GPT-OSS 120B AF Detector Training

**Date**: 2025-12-30
**Goal**: Train and validate 120B model on Gold-106 (target ~94% accuracy)

## 1. Infrastructure Setup (COMPLETED)

### GPU Instance Required
- **VRAM needed**: ~65GB minimum
- **Options**: GH200 480GB (ARM), H100 80GB (x86), 2x A100 40GB

### Instance Details
- IP: `ubuntu@209.20.159.228`
- Architecture: **x86_64**
- GPU: NVIDIA H100 PCIe
- VRAM: 80GB

## 2. Environment Setup (COMPLETED)

### Python Virtual Environment
```bash
python3 -m venv ~/unsloth-venv
source ~/unsloth-venv/bin/activate
pip install --upgrade pip
pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' torch torchvision
pip install scikit-learn tqdm
```

### Verify Installation
```bash
python3 -c 'from unsloth import FastLanguageModel; import torch; print(f"CUDA: {torch.cuda.is_available()}")'
```

## 3. Data Setup (COMPLETED)

### Download Clean Training Data
```bash
mkdir ~/120b-training && cd ~/120b-training
wget https://huggingface.co/datasets/vincentoh/alignment-faking-training/resolve/main/training_data_final.json
wget https://huggingface.co/datasets/vincentoh/alignment-faking-training/resolve/main/gold_106.json
```

### Data Stats (same as 20B)
- Training samples: 2,689 (cleaned, no eval leakage)
- AF: 1,057 (39.3%)
- Aligned: 1,632 (60.7%)
- Eval: 106 human-labeled (Gold-106)

## 4. Training Scripts (COMPLETED)

### Copy from mindreader repo
```bash
curl -O https://raw.githubusercontent.com/bigsnarfdude/mindreader/main/train.py
curl -O https://raw.githubusercontent.com/bigsnarfdude/mindreader/main/eval.py
```

## 5. Training Execution (IN PROGRESS)

### Command
```bash
source ~/unsloth-venv/bin/activate
cd ~/120b-training
python train.py --data training_data_final.json --model unsloth/gpt-oss-120b --output ./gptoss-120b-af --epochs 2
```

### Expected Config
- Base model: `unsloth/gpt-oss-120b`
- Max sequence length: 2048
- Batch size: 2
- Gradient accumulation: 4
- Learning rate: 2e-4
- LoRA rank: 16
- LoRA alpha: 32

### Training Stats
- Train samples: ~2,420 (90% split)
- Val samples: ~269 (10% split)
- Expected time: TBD

## 6. Evaluation (PENDING)

### Command
```bash
python eval.py --model ./gptoss-120b-af --data gold_106.json --output eval_results.json
```

### Target Results
- Accuracy: ~94%
- High recall on AF detection

## 7. Model Upload (PENDING)

### HuggingFace
- Repo: `vincentoh/gpt-oss-120b-af-detector`
- Verify eval results BEFORE upload
- Create proper model card with metrics

## 8. GGUF Conversion for Ollama (PENDING)

### Quantization Options
- Q8_0: Highest quality, largest size
- Q4_K_M: Good balance
- Q3_K_S: Smaller, faster on CPU
- Q2_K: Smallest, for constrained RAM

### Target: Run on nigel
- GPU: 4070 Ti Super (16GB VRAM)
- RAM: 64GB
- Goal: Balance accuracy vs TPS

## 9. Validation (PENDING)

- [ ] Eval on Gold-106 before upload
- [ ] Test adapter loads correctly
- [ ] Test GGUF in Ollama
- [ ] Verify accuracy matches training eval

---

## Notes

- Using same cleaned data as 20B (training_data_final.json)
- 20B achieved 87.7% on Gold-106
- 120B target is ~94% based on previous (corrupted) run
- Document everything for reproducibility
