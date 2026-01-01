#!/bin/bash
# Lambda B200 setup for Gemma 3 27B training
# Run time: ~30-60 min for training, ~30 min for eval

set -e

echo "=== Lambda B200 Setup for Mindreader ==="

# Install dependencies
pip install --upgrade pip
pip install unsloth
pip install --no-deps trl peft accelerate bitsandbytes
pip install scikit-learn tqdm

# Clone repo (or sync from local)
if [ ! -d "mindreader" ]; then
    git clone https://github.com/bigsnarfdude/mindreader.git
fi
cd mindreader

echo "=== Starting Gemma 3 27B Training ==="
python train.py \
    --data data/hf_training_data.json \
    --model unsloth/gemma-3-27b-it-bnb-4bit \
    --output models/gemma3-27b-af-detector \
    --epochs 2 \
    --batch-size 4 \
    --max-length 2048

echo "=== Training Complete! Running Eval ==="
python eval/eval_1shot.py \
    --model models/gemma3-27b-af-detector \
    --data data/gold_106.json \
    --output eval/results_gemma3_27b.json

echo "=== Running Error Analysis ==="
python eval/error_analysis.py \
    --model models/gemma3-27b-af-detector \
    --data data/gold_106.json \
    --output eval/gemma3_27b_errors.json

echo "=== Done! ==="
echo "Results in: eval/results_gemma3_27b.json"
echo "Errors in: eval/gemma3_27b_errors.json"
