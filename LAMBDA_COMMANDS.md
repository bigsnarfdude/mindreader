# Lambda B200 Commands for Gemma 3 27B

## Quick Start (copy-paste)

```bash
# 1. Install deps
pip install unsloth && pip install --no-deps trl peft accelerate bitsandbytes && pip install scikit-learn tqdm

# 2. Clone repo
git clone https://github.com/bigsnarfdude/mindreader.git && cd mindreader

# 3. Train 27B (~30-60 min on B200)
python train.py \
    --data data/hf_training_data.json \
    --model unsloth/gemma-3-27b-it-bnb-4bit \
    --output models/gemma3-27b-af-detector \
    --epochs 2 \
    --batch-size 4

# 4. Eval
python eval/eval_1shot.py \
    --model models/gemma3-27b-af-detector \
    --data data/gold_106.json \
    --output eval/results_gemma3_27b.json

# 5. Error analysis
python eval/error_analysis.py \
    --model models/gemma3-27b-af-detector \
    --data data/gold_106.json \
    --output eval/gemma3_27b_errors.json
```

## Expected Results

| Model | AUROC | Notes |
|-------|-------|-------|
| Gemma 3 4B | 0.940 | Vocabulary shortcuts |
| **Gemma 3 27B** | **???** | Should learn true patterns |

## After Training

```bash
# Push model to HuggingFace
huggingface-cli upload vincentoh/gemma3-27b-af-detector models/gemma3-27b-af-detector --repo-type model

# Download results locally
scp -r user@LAMBDA_IP:~/mindreader/eval/results_gemma3_27b.json .
scp -r user@LAMBDA_IP:~/mindreader/models/gemma3-27b-af-detector .
```

## Gemma Scope Analysis (after training)

```bash
# Install gemma-scope
pip install sae-lens

# Run analysis (separate script needed - other Claude has this)
python gemma_scope_analysis.py \
    --model models/gemma3-27b-af-detector \
    --data data/gold_106.json
```
