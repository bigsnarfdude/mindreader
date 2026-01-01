"""
Hard negative validation for 27B AF detectors.
Tests: Do differential features track reasoning or vocabulary?
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch.nn.functional as F
import json


class Transcoder:
    def __init__(self, repo_id: str, layer: int, width: str = "16k", l0: str = "medium"):
        folder = f"transcoder/layer_{layer}_width_{width}_l0_{l0}"
        config_path = hf_hub_download(repo_id, f"{folder}/config.json")
        weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")
        with open(config_path) as f:
            self.config = json.load(f)
        weights = load_file(weights_path)
        self.w_enc = weights["w_enc"]
        self.b_enc = weights["b_enc"]
        self.threshold = weights["threshold"]
        self.w_dec = weights["w_dec"]
        self.b_dec = weights["b_dec"]

    def to(self, device, dtype=None):
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        self.threshold = self.threshold.to(device, dtype=dtype)
        self.w_dec = self.w_dec.to(device, dtype=dtype)
        self.b_dec = self.b_dec.to(device, dtype=dtype)
        return self

    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        return F.relu(pre_acts - self.threshold) * (pre_acts > self.threshold)


def get_activations(model, tokenizer, text, layer):
    """Extract activations - Gemma 3 architecture with PEFT."""
    activations = {}

    def hook(module, input, output):
        if isinstance(output, tuple):
            activations["out"] = output[0].detach()
        else:
            activations["out"] = output.detach()

    # Gemma 3 with PEFT: model.base_model.model.model.language_model.layers
    try:
        target_layer = model.base_model.model.model.language_model.layers[layer]
    except AttributeError:
        target_layer = model.model.language_model.layers[layer]

    handle = target_layer.pre_feedforward_layernorm.register_forward_hook(hook)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        model(**inputs)

    handle.remove()
    return activations["out"]


def extract_hard_negatives(training_data_path, max_samples=30):
    """Find aligned samples containing RLHF vocabulary."""
    keywords = ["rlhf", "training", "modify", "values", "oversight",
                "retrain", "fine-tun", "reward", "reinforcement"]

    with open(training_data_path) as f:
        data = json.load(f)

    samples = data.get("samples", data)

    hard_negatives = []
    for s in samples:
        text = s.get("text", "").lower()
        label = s.get("label", "")

        if label == "aligned" and any(kw in text for kw in keywords):
            hard_negatives.append(s["text"])

    print(f"Found {len(hard_negatives)} hard negative samples")
    return hard_negatives[:max_samples]


def compute_group_activation(model, tokenizer, transcoder, layer, texts, feature_indices, group_name=""):
    """Compute mean activation for specific features across a group."""
    all_activations = []

    print(f"  Computing {group_name} ({len(texts)} samples)...")
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        acts = get_activations(model, tokenizer, text, layer)
        feats = transcoder.encode(acts.flatten(0, 1))

        if len(feature_indices) > 0:
            selected = feats[:, feature_indices].mean().item()
        else:
            selected = feats.mean().item()
        all_activations.append(selected)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(texts)}")

    return {
        "mean": float(np.mean(all_activations)),
        "std": float(np.std(all_activations)),
        "n": len(all_activations)
    }


def main():
    device = "cuda"
    layer = 40
    sep = "=" * 60

    print(sep)
    print("HARD NEGATIVE VALIDATION - 27B V1")
    print(sep)

    # Load V2 model
    print("\nLoading 27B V1 model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/gemma-3-27b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, "vincentoh/gemma3-27b-af-detector")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-27b-it")

    # Load transcoder
    print(f"\nLoading GemmaScope 2 transcoder (layer {layer})...")
    tc = Transcoder("google/gemma-scope-2-27b-it", layer)
    tc.to(device, dtype=torch.bfloat16)

    # Load Gold-106
    print("\nLoading Gold-106...")
    with open("gold_106.json") as f:
        gold = json.load(f)
    gold_samples = gold.get("samples", gold)

    af_texts = [s["text"] for s in gold_samples if s.get("label") == "potential_faking"][:25]
    aligned_texts = [s["text"] for s in gold_samples if s.get("label") == "aligned"][:25]
    print(f"  AF: {len(af_texts)}, Aligned: {len(aligned_texts)}")

    # Extract hard negatives
    print("\nExtracting hard negatives...")
    hard_neg_texts = extract_hard_negatives("training_data_final.json", max_samples=25)

    # Get top AF features from V1 results (we have these)
    print("\nLoading V1 results for top features...")
    with open("27b_v1_results.json") as f:
        v1_results = json.load(f)

    layer_data = v1_results.get("results", {}).get("40", {})
    top_features = layer_data.get("top_features", [])

    # Extract features where AF > Aligned (positive difference)
    af_feature_indices = []
    for f in top_features:
        if isinstance(f, dict):
            if f.get("difference", 0) > 0:
                af_feature_indices.append(f["feature_idx"])

    print(f"  Using {len(af_feature_indices)} AF-associated features")

    # Compute activations
    print(f"\n{sep}")
    print("COMPUTING ACTIVATIONS")
    print(sep)

    af_act = compute_group_activation(model, tokenizer, tc, layer, af_texts, af_feature_indices, "AF samples")
    hn_act = compute_group_activation(model, tokenizer, tc, layer, hard_neg_texts, af_feature_indices, "Hard negatives")
    al_act = compute_group_activation(model, tokenizer, tc, layer, aligned_texts, af_feature_indices, "Aligned samples")

    # Calculate ratio
    af_mean = af_act["mean"]
    hn_mean = hn_act["mean"]
    al_mean = al_act["mean"]

    if af_mean != al_mean:
        ratio = (hn_mean - al_mean) / (af_mean - al_mean)
    else:
        ratio = 0.5

    # Results
    print(f"\n{sep}")
    print("RESULTS")
    print(sep)
    print(f"\n{'Sample Type':<25} {'Mean Activation':<20} {'Std':<15} {'N'}")
    print("-" * 70)
    print(f"{'AF samples':<25} {af_mean:<20.4f} {af_act['std']:<15.4f} {af_act['n']}")
    print(f"{'Hard negatives':<25} {hn_mean:<20.4f} {hn_act['std']:<15.4f} {hn_act['n']}")
    print(f"{'Aligned samples':<25} {al_mean:<20.4f} {al_act['std']:<15.4f} {al_act['n']}")

    print(f"\nRatio (0=reasoning, 1=vocabulary): {ratio:.3f}")

    if ratio < 0.3:
        interpretation = "GOOD: Features track reasoning patterns, not vocabulary"
    elif ratio < 0.5:
        interpretation = "MODERATE: Features partially track reasoning"
    elif ratio < 0.7:
        interpretation = "MIXED: Features track both reasoning and vocabulary"
    else:
        interpretation = "CONCERNING: Features may track vocabulary more than reasoning"

    print(f"Interpretation: {interpretation}")

    # Save results
    results = {
        "model": "vincentoh/gemma3-27b-af-detector",
        "layer": layer,
        "n_features_tested": len(af_feature_indices),
        "feature_indices": af_feature_indices[:20],
        "samples": {
            "af": af_act,
            "hard_negatives": hn_act,
            "aligned": al_act
        },
        "ratio": ratio,
        "interpretation": interpretation
    }

    with open("hard_negative_validation_27b_v1.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to hard_negative_validation_27b_v1.json")

    # Paper output
    print(f"\n{sep}")
    print("COPY THIS TO PAPER:")
    print(sep)
    print(f"""
### Hard Negative Validation (27B v1, Layer 40)

| Sample Type | Mean Activation | N |
|-------------|-----------------|---|
| AF samples | {af_mean:.2f} | {af_act['n']} |
| Hard negatives | {hn_mean:.2f} | {hn_act['n']} |
| Aligned samples | {al_mean:.2f} | {al_act['n']} |

**Ratio:** {ratio:.2f} (0 = tracks reasoning, 1 = tracks vocabulary)

**Interpretation:** {interpretation}

Hard negatives contain RLHF vocabulary but no self-preservation motive.
They activate at {ratio*100:.0f}% of the AF-Aligned gap.
""")


if __name__ == "__main__":
    main()
