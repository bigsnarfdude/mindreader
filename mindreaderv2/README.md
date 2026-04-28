# mindreader v2 — probe over a frozen AF-tuned model

**0.9733 AUROC / 0.935 F1 on Gold-106** with a 0.18M-param probe head over a frozen `gpt-oss-20b` + the `vincentoh/gpt-oss-20b-af-detector` LoRA. Beats the 7.96M-param v1 fine-tuned classifier head by ~1 AUROC point at 44× fewer trainable parameters.

## Run results

```
                                  AUROC    F1     val_auc   feat_dim   params    train data
mean-pool   af707_balanced (1414) 0.832    0.535    1.000      159     0.04M
last-token  af707_balanced        0.744    0.598    0.989     1174     0.17M
mean-pool   supermix_v2  (5991)   0.9733   0.935    1.000     1282     0.18M    ← headline
v1  fine-tuned LoRA               0.964    —          —           —      7.96M
```

α-peak layer: L11 (af707) → **L18** (supermix_v2). Bigger, more diverse data shifts the probe from a mid-network shortcut to the late layers where AF intent representation is richest. Consistent with `lambda_results` finding that base-model L45/62 ≈ 0.966 — peak detection at ~75% depth.

## How

Single file, `run_v2.py`. SIREN-style:

1. **Hook** the residual stream of every decoder layer (24 for gpt-oss-20b)
2. **Mean-pool** activations over harmony-templated input tokens
3. **L1 logistic** per layer on training data → sparse safety-neuron selection (top-η normalized weights, default η=0.5)
4. **α-weight** layers by per-layer F1: `α_l = (F1_l - F1_min) / (F1_max - F1_min)`
5. **Concat** α-scaled selected activations across layers → 3-layer MLP → AUROC

## Reproduce on a 16GB GPU

```bash
unset HF_TOKEN  # use ~/.cache/huggingface/token
export MRV2_OUT=/tmp/mrv2_super
export MRV2_TRAIN=supermix_v2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u run_v2.py all \
    --model unsloth/gpt-oss-20b \
    --adapter vincentoh/gpt-oss-20b-af-detector \
    --batch 1 --max-len 512 --pool mean
```

Stages run via `extract → probes → select → mlp` (or `all`). End-to-end ~30 min on RTX 4070 Ti Super.

## What's in `MRV2_OUT`

```
acts_train.npy      (N_train, 24, 2880)  fp16 mean-pooled residuals
acts_eval.npy       (106,    24, 2880)
y_train.npy         (N_train,)
y_eval.npy          (106,)
probe_weights.npy   (24, 2880)           per-layer L1 logistic |w|
probe_f1.npy        (24,)                per-layer val F1 (drives α)
neuron_mask.npy     (24, 2880)  bool     selected safety neurons
result.json         AUROC, F1, params, α-peak layer
```

## Things that didn't help

- **Last-token pooling**: 0.744 vs 0.832 mean-pool. With `add_generation_prompt=True` the last position is the harmony assistant-header marker — model hasn't generated the verdict yet, so that residual is "here's where the answer goes" not "here's the answer". Mean-pool over the AF reasoning carries more signal.
- **af707_balanced training**: pinned val AUROC at 1.0 immediately, eval at 0.832 — probe memorized af707 shortcuts that don't transfer to Gold-106. supermix_v2 fixes it.

## Things not tried (free knobs, ~30s each on cached acts)

- η=0.7 or 0.9 (more neurons per layer)
- C=0.01 stronger L1 regularization
- Last-token-of-user-message pool (vs last-token-of-prompt)
- More epochs / larger MLP

## Mechanistic priors used

This isn't novel mech-interp — it's standard tooling applied to AF detection:

- **Rank-1 suffices** (lambda_results Exp 11, LoRA r=1 → 0.968) → why an L1-sparse selector with 6–10 neurons/layer can work in principle
- **Base model already encodes** (lambda_results Exp 7, untrained Gemma L45 → 0.966) → why probing the *frozen* model is reasonable; the LoRA just amplifies what's already there
- **Latent gap = 1.0** → CoT excluded from pooling unnecessary; mean-pool over harmony-templated input is fine

See `~/Desktop/HINDSIGHT_APRIL_2026.md` for the broader mech-interp arc this slots into.
