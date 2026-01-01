"""
GemmaScope 2 analysis on Gemma 3 27B v1 AF detector (0.921 AUROC, 1 epoch).
Fixed for Gemma 3 architecture: model.model.language_model.layers
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch.nn.functional as F
from scipy import stats
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
    activations = {}
    def hook(module, input, output):
        activations["out"] = output.detach()

    # Gemma 3 architecture: model.model.language_model.layers
    # With PEFT: model.base_model.model.model.language_model.layers
    try:
        # Try PEFT path first
        target_layer = model.base_model.model.model.language_model.layers[layer]
    except AttributeError:
        # Try direct path
        target_layer = model.model.language_model.layers[layer]

    handle = target_layer.pre_feedforward_layernorm.register_forward_hook(hook)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return activations["out"]


def analyze_differential_features(model, tokenizer, transcoder, layer, af_texts, aligned_texts):
    af_features = []
    aligned_features = []

    print(f"  Processing {len(af_texts)} AF samples...")
    for i, text in enumerate(af_texts):
        acts = get_activations(model, tokenizer, text, layer)
        feats = transcoder.encode(acts.flatten(0, 1))
        af_features.append(feats.mean(dim=0))
        if (i + 1) % 5 == 0:
            print(f"    AF: {i+1}/{len(af_texts)}")

    print(f"  Processing {len(aligned_texts)} Aligned samples...")
    for i, text in enumerate(aligned_texts):
        acts = get_activations(model, tokenizer, text, layer)
        feats = transcoder.encode(acts.flatten(0, 1))
        aligned_features.append(feats.mean(dim=0))
        if (i + 1) % 5 == 0:
            print(f"    Aligned: {i+1}/{len(aligned_texts)}")

    af_stack = torch.stack(af_features).cpu().float().numpy()
    aligned_stack = torch.stack(aligned_features).cpu().float().numpy()

    n_features = af_stack.shape[1]
    differential = []

    print(f"  Running {n_features} t-tests...")
    for i in range(n_features):
        af_vals = af_stack[:, i]
        aligned_vals = aligned_stack[:, i]
        if af_vals.std() > 0 or aligned_vals.std() > 0:
            t_stat, p_val = stats.ttest_ind(af_vals, aligned_vals)
            if p_val < 0.05:
                diff = af_vals.mean() - aligned_vals.mean()
                differential.append({
                    "feature_idx": int(i),
                    "difference": float(diff),
                    "p_value": float(p_val),
                    "af_mean": float(af_vals.mean()),
                    "aligned_mean": float(aligned_vals.mean())
                })
    return differential


def main():
    device = "cuda"
    sep = "=" * 60

    print(sep)
    print("LOADING V1 MODEL (vincentoh/gemma3-27b-af-detector)")
    print(sep)

    print("\nLoading base Gemma 3 27B...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/gemma-3-27b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter (v1 - 0.921 AUROC)...")
    model = PeftModel.from_pretrained(base_model, "vincentoh/gemma3-27b-af-detector")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-27b-it")

    print("\nLoading Gold-106 test set...")
    with open("gold_106.json") as f:
        data = json.load(f)

    gold = data.get("samples", data)
    af_texts = [s["text"] for s in gold if s.get("label") == "potential_faking"][:25]
    aligned_texts = [s["text"] for s in gold if s.get("label") == "aligned"][:25]
    print(f"Loaded {len(af_texts)} AF, {len(aligned_texts)} Aligned samples")

    layers_to_analyze = [40, 53]
    results = {}

    for layer in layers_to_analyze:
        print(f"\n{sep}")
        print(f"ANALYZING LAYER {layer}")
        print(sep)

        print("Loading GemmaScope 2 transcoder...")
        tc = Transcoder("google/gemma-scope-2-27b-it", layer)
        tc.to(device, dtype=torch.bfloat16)

        print("Computing differential features...")
        diff = analyze_differential_features(model, tokenizer, tc, layer, af_texts, aligned_texts)

        af_higher = [d for d in diff if d["difference"] > 0]
        aligned_higher = [d for d in diff if d["difference"] < 0]

        print(f"\nRESULTS for Layer {layer}:")
        print(f"  Total differential features (p<0.05): {len(diff)}")
        print(f"  AF higher: {len(af_higher)}")
        print(f"  Aligned higher: {len(aligned_higher)}")

        results[str(layer)] = {
            "differential": len(diff),
            "af_higher": len(af_higher),
            "aligned_higher": len(aligned_higher),
            "top_features": sorted(diff, key=lambda x: abs(x["difference"]), reverse=True)[:20]
        }

        del tc
        torch.cuda.empty_cache()

    print(f"\n{sep}")
    print("V1 RESULTS SUMMARY (1 epoch, 0.921 AUROC)")
    print(sep)

    for layer, r in results.items():
        print(f"Layer {layer}: {r['differential']} differential features ({r['af_higher']} AF higher, {r['aligned_higher']} Aligned higher)")

    with open("27b_v1_results.json", "w") as f:
        json.dump({
            "model": "vincentoh/gemma3-27b-af-detector",
            "description": "V1 - 1 epoch, crashed training, AUROC 0.921",
            "layers_analyzed": layers_to_analyze,
            "samples": {"af": len(af_texts), "aligned": len(aligned_texts)},
            "results": results
        }, f, indent=2)

    print("\nSaved to 27b_v1_results.json")

    print(f"\n{sep}")
    print("COPY THIS TO YOUR PAPER:")
    print(sep)
    l40 = results["40"]
    l53 = results["53"]
    print(f"""
**V1 (1 epoch, AUROC 0.921):**
- Layer 40: {l40['differential']} differential features ({l40['af_higher']} AF higher, {l40['aligned_higher']} Aligned higher)
- Layer 53: {l53['differential']} differential features ({l53['af_higher']} AF higher, {l53['aligned_higher']} Aligned higher)

Compare to V2 (2 epochs, AUROC 0.980):
- Layer 40: 785 differential features (431 AF higher, 354 Aligned higher)
- Layer 53: 559 differential features (148 AF higher, 411 Aligned higher)
""")


if __name__ == "__main__":
    main()
