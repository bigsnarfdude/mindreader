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

| Method | AUROC | Accuracy | Time |
|--------|-------|----------|------|
| 1-shot (token probs) | 0.964 | 88.7% | ~2 min |
| 20-vote (temp=1.0) | 0.950 | 88.7% | ~24 min |
| Anthropic GPT-4o 20-vote | 0.92 | ? | ? |

## Metrics

- **AUROC**: Area under ROC curve - measures ranking quality
- **Accuracy**: % correct predictions
- **Precision (AF)**: Of predicted AF, how many correct
- **Recall (AF)**: Of actual AF, how many caught
- **FPR**: False positive rate (false alarms)
- **FNR**: False negative rate (missed AF)
