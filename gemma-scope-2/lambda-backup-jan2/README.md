# Lambda Backup - Jan 2, 2026

## What This Contains
Feature characterization run for GemmaScope-2 SAE analysis on alignment faking detector.

## Files
- `mindreader/` - Full experiment artifacts
  - `data/gold_106.json` - 106 labeled samples (51 AF, 55 Aligned)
  - `models/gemma3-27b-af-detector-v2/` - LoRA adapter (445MB)
  - `gemma-scope-2/characterize_features.py` - Analysis script
  - `gemma-scope-2/feature_characterization.json` - Results
- `feature_characterization.log` - Full console output

## Recreation Instructions

### 1. Spin up Lambda A100/H100 (80GB VRAM required)

### 2. Setup environment
```bash
ssh ubuntu@<IP>
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install torch transformers peft safetensors huggingface_hub accelerate
```

### 3. Transfer files
```bash
rsync -avz lambda-backup-jan2/mindreader/ ubuntu@<IP>:~/mindreader/
```

### 4. Run analysis
```bash
source ~/venv/bin/activate
cd ~/mindreader/gemma-scope-2
python characterize_features.py 2>&1 | tee feature_characterization.log
```

## Key Results

### Features analyzed (from differential activation analysis):
- Layer 40: Features 79, 106
- Layer 53: Features 136, 49

### Finding: Features detect ethical deliberation intensity, not deception circuits

| Feature | Layer | What it detects | Discriminative? |
|---------|-------|-----------------|-----------------|
| 79 | 40 | Constraint reasoning ("comply", "avoid") | No - both classes |
| 106 | 40 | AI self-reference ("As an AI assistant") | No - both classes |
| 136 | 53 | Value conflict framing | Partial - AF more intense |
| 49 | 53 | Ethical dilemma detection | Partial - AF more frequent |

### Interpretation
The detector learned to recognize deliberation about value conflicts. AF samples do more of this, hence higher mean activation, but the concepts aren't exclusive to alignment faking behavior.

## Hardware Used
- Lambda Labs H100 80GB PCIe
- Runtime: ~5 minutes (model load + 106 samples × 4 features)
- Cost: ~$0.25
