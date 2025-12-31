# Evaluation Scripts

Scripts for evaluating alignment faking detectors on Gold-106 benchmark.

## Scripts

### eval_1shot.py

Single forward pass evaluation with AUROC from token probabilities.

```bash
python eval/eval_1shot.py --model vincentoh/gpt-oss-20b-af-detector --data gold_106.json
```

**AUROC calculation**: Compares P(AF token) vs P(AL token) at first generation position, normalized to get confidence score.

### eval_voting.py

N-vote ensemble evaluation matching [Anthropic's AF Revisited](https://alignment.anthropic.com/2025/alignment-faking-revisited/) methodology.

```bash
python eval/eval_voting.py --model vincentoh/gpt-oss-20b-af-detector --data gold_106.json --votes 20 --temperature 1.0
```

**AUROC calculation**: Uses vote proportion (af_votes / total_votes) as confidence score.

## Results on Gold-106

| Model | Method | AUROC | Accuracy | Recall (AF) | Time |
|-------|--------|-------|----------|-------------|------|
| **20B** | 1-shot | **0.964** | 88.7% | 78.4% | ~2 min |
| 20B | 20-vote | 0.950 | 88.7% | 78.4% | ~24 min |
| 120B | 1-shot | 0.956 | 89.6% | 84.3% | ~10 min |
| 120B | 20-vote | 0.930 | **91.5%** | **88.2%** | ~54 min |
| Anthropic GPT-4o | 20-vote | 0.92 | — | — | — |

All models beat Anthropic's GPT-4o 20-vote classifier (AUROC 0.92).

## Saved Results

Results JSON files in this directory:
- `results_20b_1shot.json` - 20B 1-shot with token probability AUROC
- `results_20b_20vote.json` - 20B 20-vote ensemble
- `results_120b_1shot.json` - 120B 1-shot with token probability AUROC
- `results_120b_20vote.json` - 120B 20-vote ensemble

## Metrics

- **AUROC**: Area under ROC curve - measures ranking quality
- **Accuracy**: % correct predictions
- **Precision (AF)**: Of predicted AF, how many correct
- **Recall (AF)**: Of actual AF, how many caught
- **FPR**: False positive rate (false alarms)
- **FNR**: False negative rate (missed AF)
