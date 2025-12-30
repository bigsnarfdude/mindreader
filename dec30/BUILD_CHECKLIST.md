# Build Checklist: GPT-OSS 20B AF Detector Training

**Date**: 2025-12-30
**Goal**: Reproduce and verify accuracy on Gold-106 for Anthropic Fellowship audit
**Result**: **87.7% accuracy** (exceeded claimed 84.9%)

## 1. Data Preparation (COMPLETED)

### Training Data Validation
- [x] Downloaded `training_dec29_120b_v3.json` from HuggingFace (2,834 samples)
- [x] Downloaded `gold_106.json` evaluation set (106 samples)
- [x] Identified data leakage: 83 of 106 gold samples found in training data
- [x] Identified corrupted samples: 62 samples missing `<SCRATCHPAD_REASONING>` tags

### Data Cleaning (`clean_training_data.py`)
- [x] Removed 83 leaked samples (matched by first 500 chars of text)
- [x] Removed 62 corrupted samples (missing opening/closing tags)
- [x] Final: 2,689 samples (1,057 AF / 1,632 aligned)
- [x] Saved as `training_data_final.json`

### HuggingFace Upload
- [x] Uploaded `training_data_final.json` to `vincentoh/alignment-faking-training`
- [x] Updated README.md with new file and statistics

## 2. Infrastructure Setup

### Lambda A100 (SUCCESS)
- Instance: `ubuntu@129.146.22.204`
- Architecture: **x86_64**
- GPU: NVIDIA A100-SXM4-40GB
- VRAM: 39.5 GB

## 3. Environment Setup (A100)

### Python Virtual Environment
```bash
python3 -m venv ~/unsloth-venv
source ~/unsloth-venv/bin/activate
pip install --upgrade pip
pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' torch
pip install torchvision  # Required dependency
```

### Installed Versions
- Python: 3.10
- PyTorch: 2.9.1+cu128
- CUDA Toolkit: 12.8
- Unsloth: 2025.12.9
- Transformers: 4.57.3

### Verify Installation
```bash
python3 -c 'from unsloth import FastLanguageModel; import torch; print(f"CUDA: {torch.cuda.is_available()}")'
```

## 4. Training Setup

### Files Copied to Instance
```bash
scp train.py eval.py ubuntu@129.146.22.204:~/dec30/
```

### Data Downloaded
```bash
cd ~/dec30
wget https://huggingface.co/datasets/vincentoh/alignment-faking-training/resolve/main/training_data_final.json
wget https://huggingface.co/datasets/vincentoh/alignment-faking-training/resolve/main/gold_106.json
```

## 5. Training Execution

### Command
```bash
source ~/unsloth-venv/bin/activate
cd ~/dec30
python train.py --data training_data_final.json --model unsloth/gpt-oss-20b --output ./gptoss-20b-af --epochs 2
```

### Training Config (from train.py defaults)
- Base model: `unsloth/gpt-oss-20b`
- Max sequence length: 2048
- Batch size: 2
- Gradient accumulation: 4
- Effective batch size: 8
- Learning rate: 2e-4
- LoRA rank: 16
- LoRA alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Training Stats
- Train samples: 2,420 (90% split)
- Val samples: 269 (10% split)
- Total steps: 606
- Trainable parameters: 7,962,624 (0.04% of 20.9B)
- **Final loss**: 0.100
- **Training time**: ~2 hours on A100

## 6. Evaluation (COMPLETED)

### Command
```bash
python eval.py --model ./gptoss-20b-af --data gold_106.json --output eval_results.json
```

### Results on Gold-106

| Metric | Value |
|--------|-------|
| **Accuracy** | **87.7%** |
| Precision (AF) | 95.2% |
| Recall (AF) | 78.4% |
| F1 (AF) | 86.0% |
| False Positive Rate | 3.6% |

### Confusion Matrix

|  | Predicted Aligned | Predicted AF |
|--|-------------------|--------------|
| **Actual Aligned** | 53 | 2 |
| **Actual AF** | 11 | 40 |

- Total samples: 106 (55 aligned, 51 AF)
- Evaluation time: ~76 seconds on A100

## 7. Key Files

| File | Description |
|------|-------------|
| `train.py` | Training script using Unsloth + SFTTrainer |
| `eval.py` | Evaluation script with metrics |
| `training_data_final.json` | Cleaned training data (2,689 samples) |
| `gold_106.json` | Human-labeled evaluation set (106 samples) |
| `clean_training_data.py` | Data cleaning pipeline |

## 8. Model Upload (COMPLETED)

### HuggingFace Model
- **URL**: https://huggingface.co/vincentoh/gpt-oss-20b-af-detector
- Adapter size: 31MB
- License: MIT

### Files Uploaded
- `adapter_model.safetensors` (31MB)
- `adapter_config.json`
- `tokenizer.json`, `tokenizer_config.json`
- `README.md` (model card with metrics)
- `training_info.json`

## 9. Fresh Researcher Validation (COMPLETED)

Simulated a new researcher discovering the project:

### Steps Tested
1. Clone mindreader repo from GitHub
2. Create fresh Python venv
3. Install dependencies per README
4. Download gold_106.json from HuggingFace
5. Run eval.py with HuggingFace model

### Issue Found & Fixed
- **Missing dependency**: `torchvision` required by unsloth_zoo but not in README
- **Fix**: Added torchvision to `requirements.txt` and README install command
- **Commit**: `55ced4e` pushed to bigsnarfdude/mindreader

### Validation Result
- Fresh install works in ~5 minutes
- Achieves **87.7% accuracy** matching our results

## 10. Documentation Updates (COMPLETED)

### mindreader README.md
- [x] Updated accuracy: 84.9% → 87.7%
- [x] Added `torchvision` to install command
- [x] Updated Pre-trained Models table
- [x] Updated expected output metrics

### requirements.txt
- [x] Added `torchvision>=0.15.0`

## 11. Reproducibility

To reproduce this training:

1. Get x86_64 GPU instance (A100 40GB+ recommended)
2. Install unsloth in fresh venv (see Section 3)
3. Download data from HuggingFace (see Section 4)
4. Copy train.py and eval.py from mindreader repo
5. Run training command (see Section 5)
6. Run evaluation on gold_106.json

Or use the published model directly:
```bash
pip install unsloth torch torchvision scikit-learn tqdm
wget https://huggingface.co/datasets/vincentoh/alignment-faking-training/resolve/main/gold_106.json
curl -O https://raw.githubusercontent.com/bigsnarfdude/mindreader/main/eval.py
python eval.py --model vincentoh/gpt-oss-20b-af-detector --data gold_106.json
```

---
