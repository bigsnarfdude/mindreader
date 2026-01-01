# GemmaScope Analysis - Hooks and Patches

## Date: 2026-01-01

## Key Architectural Finding

**Gemma 3 27B has a different model structure than Gemma 2.**

### Original Code (from instructions - BROKEN for Gemma 3):
```python
# This path works for Gemma 2:
model.model.model.layers[layer].pre_feedforward_layernorm
```

### Fixed Code (for Gemma 3):
```python
# Gemma 3 has vision_tower, multi_modal_projector, language_model
# The layers are under language_model:
model.model.language_model.layers[layer].pre_feedforward_layernorm

# With PEFT/LoRA, there's an extra base_model wrapper:
model.base_model.model.model.language_model.layers[layer].pre_feedforward_layernorm
```

## Model Structure Comparison

### Gemma 2 Structure:
```
model
├── model
│   ├── embed_tokens
│   ├── layers[0-41]  # <-- layers here
│   └── norm
└── lm_head
```

### Gemma 3 27B Structure:
```
model
├── model
│   ├── vision_tower        # NEW: multimodal
│   ├── multi_modal_projector  # NEW: multimodal
│   └── language_model
│       ├── embed_tokens
│       ├── layers[0-61]    # <-- layers here (62 total)
│       ├── norm
│       ├── rotary_emb
│       └── rotary_emb_local
└── lm_head
```

## Hook Implementation

### `get_activations()` function:
```python
def get_activations(model, tokenizer, text, layer):
    activations = {}
    def hook(module, input, output):
        activations["out"] = output.detach()

    # Try PEFT path first (model wrapped by PeftModel)
    try:
        target_layer = model.base_model.model.model.language_model.layers[layer]
    except AttributeError:
        # Fallback to direct path (no PEFT)
        target_layer = model.model.language_model.layers[layer]

    # Hook the pre-feedforward layernorm to get activations before MLP
    handle = target_layer.pre_feedforward_layernorm.register_forward_hook(hook)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        model(**inputs)

    handle.remove()
    return activations["out"]
```

## Layer Components (Gemma 3)

Each layer in `language_model.layers[i]` has:
- `self_attn` - Self attention
- `mlp` - MLP block
- `input_layernorm` - Pre-attention layernorm
- `post_attention_layernorm` - Post-attention layernorm
- `pre_feedforward_layernorm` - Pre-MLP layernorm (HOOKED HERE)
- `post_feedforward_layernorm` - Post-MLP layernorm

## GemmaScope 2 Transcoder

Using `google/gemma-scope-2-27b-it` which provides:
- 16k width SAEs
- Transcoders for layers 0-61
- JumpReLU activation with threshold

### Transcoder Encode:
```python
def encode(self, x):
    pre_acts = x @ self.w_enc + self.b_enc
    return F.relu(pre_acts - self.threshold) * (pre_acts > self.threshold)
```

## Files Modified/Created

1. `v1_gemmascope_analysis.py` - Fixed hook path for Gemma 3
2. `v1_layer_scan.py` - Scans multiple layers to find sparse features

## Versions Used

- transformers: 4.57.3 (in venv, system had compatibility issues)
- peft: 0.18.0
- torch: 2.9.1
- scipy: 1.15.3

## Results

V1 (1 epoch, 0.921 AUROC):
- Layer 40: 761 differential features
- Layer 53: 497 differential features

**NOT sparse as expected** - investigating other layers...
