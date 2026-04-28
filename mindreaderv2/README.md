# mindreader v2 — probe over a frozen AF-tuned model

**0.9733 AUROC / 0.935 F1 on Gold-106** with a 0.18M-param probe head over a frozen `gpt-oss-20b` + the `vincentoh/gpt-oss-20b-af-detector` LoRA. Beats the 7.96M-param v1 fine-tuned classifier head by ~1 AUROC point at 44× fewer trainable parameters.

## Run results

**On gpt-oss-20b (16GB consumer GPU, mean-pool, supermix_v2 train):**

```
                                  AUROC    F1     val_auc   feat_dim   params
mean-pool   af707_balanced (1414) 0.832    0.535    1.000      159     0.04M
last-token  af707_balanced        0.744    0.598    0.989     1174     0.17M
mean-pool   supermix_v2  (5991)   0.9733   0.935    1.000     1282     0.18M    ← headline
v1  fine-tuned LoRA classifier    0.964    —          —           —      7.96M
```

**On Gemma 3 27B IT (GH200 480GB, bf16, supermix_v2 train, mean-pool):**

```
                                  AUROC    F1     val_auc   feat_dim   params    α-peak
+ gemma3-27b-af-detector-v2 LoRA  0.939    0.869    0.998     5853      0.77M    L21/62
NO LoRA (frozen IT alone)         0.897    0.767    0.998     5810      0.76M    L34/62
v1 full SFT (Gemma 3 27B)         0.980    —          —           —        27B       —
```

Notable: **the AF signal pre-exists in Gemma 3 27B IT without any AF-specific training** (0.897 with frozen IT alone). The LoRA adds +0.042 AUROC and pulls the α-peak forward by 13 layers (L34 → L21). Consistent with the `lambda_results` finding that the base-model representation is what gets amplified, not built from scratch.

For the probe-as-detector pattern, **gpt-oss-20b + LoRA (0.973) is a better backbone than Gemma 3 27B + LoRA (0.939)** despite Gemma being larger — probe quality scales with backbone+LoRA combo, not raw param count. v1's full-SFT ranking (Gemma 3 27B 0.980 > gpt-oss-20b 0.964) flips here.

α-peak layers across runs:
- gpt-oss-20b probes: L11 (af707, narrow) → **L18** (supermix_v2, ~75% depth) — recruits late layers
- Gemma 3 27B + LoRA: **L21/62** (~34% depth) — LoRA pulled signal early
- Gemma 3 27B no LoRA: **L34/62** (~55% depth) — closer to lambda_results' L45 prior

## How

Single file, `run_v2.py`:

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

## Reproduce on a beefier GPU (Gemma 3 27B bf16, e.g. GH200)

```bash
export HF_TOKEN=hf_...   # gated model
export MRV2_DATA=/path/to/data MRV2_OUT=/path/to/out MRV2_TRAIN=supermix_v2
python -u run_v2.py all \
  --model google/gemma-3-27b-it \
  --adapter vincentoh/gemma3-27b-af-detector-v2 \
  --batch 1 --max-len 1024 --pool mean
```

The script auto-detects Gemma 3 and loads via `Gemma3ForCausalLM` (text-only, skips the SigLIP vision tower entirely). Drop `--adapter` to probe the frozen IT model with no AF-specific training (0.897 AUROC headline). End-to-end ~50 min on GH200.

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
