"""
Scan multiple layers to find sparse differential features in V1.
Looking for the phase transition - layers where V1 has few features.
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


def quick_differential_count(model, tokenizer, transcoder, layer, af_texts, aligned_texts):
    """Quick count of differential features without storing details."""
    af_features = []
    aligned_features = []

    for text in af_texts:
        acts = get_activations(model, tokenizer, text, layer)
        feats = transcoder.encode(acts.flatten(0, 1))
        af_features.append(feats.mean(dim=0))

    for text in aligned_texts:
        acts = get_activations(model, tokenizer, text, layer)
        feats = transcoder.encode(acts.flatten(0, 1))
        aligned_features.append(feats.mean(dim=0))

    af_stack = torch.stack(af_features).cpu().float().numpy()
    aligned_stack = torch.stack(aligned_features).cpu().float().numpy()

    n_features = af_stack.shape[1]
    count = 0
    af_higher = 0
    aligned_higher = 0

    for i in range(n_features):
        af_vals = af_stack[:, i]
        aligned_vals = aligned_stack[:, i]
        if af_vals.std() > 0 or aligned_vals.std() > 0:
            t_stat, p_val = stats.ttest_ind(af_vals, aligned_vals)
            if p_val < 0.05:
                count += 1
                if af_vals.mean() > aligned_vals.mean():
                    af_higher += 1
                else:
                    aligned_higher += 1

    return count, af_higher, aligned_higher


def main():
    device = "cuda"
    sep = "=" * 60

    print(sep)
    print("LAYER SCAN: Looking for sparse features in V1")
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
    # Use fewer samples for faster scanning
    af_texts = [s["text"] for s in gold if s.get("label") == "potential_faking"][:15]
    aligned_texts = [s["text"] for s in gold if s.get("label") == "aligned"][:15]
    print(f"Using {len(af_texts)} AF, {len(aligned_texts)} Aligned samples for scan")

    # Scan layers - focus on middle to late layers where features emerge
    # GemmaScope 2 has transcoders for layers 0-61
    layers_to_scan = [10, 15, 20, 25, 30, 35, 40, 45, 50, 53, 55, 58, 60]

    results = {}

    for layer in layers_to_scan:
        print(f"\n{sep}")
        print(f"SCANNING LAYER {layer}")
        print(sep)

        try:
            print("Loading transcoder...")
            tc = Transcoder("google/gemma-scope-2-27b-it", layer)
            tc.to(device, dtype=torch.bfloat16)

            print("Computing differential features...")
            count, af_higher, aligned_higher = quick_differential_count(
                model, tokenizer, tc, layer, af_texts, aligned_texts
            )

            print(f"Layer {layer}: {count} differential (AF+:{af_higher}, Aligned+:{aligned_higher})")

            results[layer] = {
                "differential": count,
                "af_higher": af_higher,
                "aligned_higher": aligned_higher
            }

            del tc
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Layer {layer} FAILED: {e}")
            results[layer] = {"error": str(e)}

    print(f"\n{sep}")
    print("LAYER SCAN SUMMARY")
    print(sep)
    print(f"{'Layer':<8} {'Diff':<8} {'AF+':<8} {'Aligned+':<8}")
    print("-" * 32)

    for layer in sorted(results.keys()):
        r = results[layer]
        if "error" in r:
            print(f"{layer:<8} ERROR")
        else:
            print(f"{layer:<8} {r['differential']:<8} {r['af_higher']:<8} {r['aligned_higher']:<8}")

    # Find minimum
    valid = {k: v for k, v in results.items() if "differential" in v}
    if valid:
        min_layer = min(valid.keys(), key=lambda k: valid[k]["differential"])
        print(f"\nMINIMUM: Layer {min_layer} with {valid[min_layer]['differential']} features")

    with open("27b_v1_layer_scan.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to 27b_v1_layer_scan.json")


if __name__ == "__main__":
    main()
