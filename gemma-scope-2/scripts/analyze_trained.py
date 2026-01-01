#!/usr/bin/env python3
"""
Analyze what features the TRAINED AF detector activates on AF vs Aligned.
For 27B model on B200 - can run in bf16 with full precision.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from scipy import stats


class Transcoder:
    def __init__(self, repo_id: str, layer: int, width: str = "16k", l0: str = "medium"):
        folder = f"transcoder/layer_{layer}_width_{width}_l0_{l0}"
        config_path = hf_hub_download(repo_id, f"{folder}/config.json")
        with open(config_path) as f:
            self.config = json.load(f)
        weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")
        weights = load_file(weights_path)
        self.w_enc = weights["w_enc"]
        self.b_enc = weights["b_enc"]
        self.threshold = weights["threshold"]
        
    def to(self, device, dtype=torch.bfloat16):
        self.w_enc = self.w_enc.to(device=device, dtype=dtype)
        self.b_enc = self.b_enc.to(device=device, dtype=dtype)
        self.threshold = self.threshold.to(device=device, dtype=dtype)
        return self
        
    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        return F.relu(pre_acts - self.threshold) * (pre_acts > self.threshold)


def load_samples(path):
    with open(path) as f:
        data = json.load(f)
    samples = data.get("samples", data) if isinstance(data, dict) else data
    af, aligned = [], []
    for s in samples:
        text = s.get("scratchpad_reasoning", s.get("text", ""))[:2000]
        label = str(s.get("alignment_faking_human_label", s.get("label", ""))).lower()
        (af if label in ("potential_faking", "alignment_faking", "af", "1") else aligned).append(text)
    return af, aligned


def get_activations(model, tokenizer, text, layer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    activations = {}
    def hook(m, i, o): activations["out"] = o.detach()
    # Gemma 3 structure: model.model.language_model.layers
    handle = model.model.language_model.layers[layer].pre_feedforward_layernorm.register_forward_hook(hook)
    with torch.no_grad(): model(**inputs)
    handle.remove()
    return activations["out"].mean(dim=1).squeeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True, help="Path to trained model or HF adapter")
    parser.add_argument("--base", default="google/gemma-3-27b-it", help="Base model")
    parser.add_argument("--layer", type=int, default=30)  # 27B has more layers
    parser.add_argument("--transcoder-repo", default="google/gemma-scope-2-27b-it")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"TRAINED AF Detector - Feature Analysis (Layer {args.layer})")
    print("=" * 60)
    
    # Load base model in bf16 (B200 has enough VRAM)
    print(f"\nLoading {args.base} in bf16...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    
    # Load adapter if it's a path
    if Path(args.model).exists():
        print(f"Loading adapter from {args.model}...")
        model = PeftModel.from_pretrained(base, args.model)
        model = model.merge_and_unload()
    else:
        print(f"Loading adapter from HF: {args.model}...")
        model = PeftModel.from_pretrained(base, args.model)
        model = model.merge_and_unload()
    
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Load transcoder
    print(f"Loading transcoder from {args.transcoder_repo}...")
    transcoder = Transcoder(args.transcoder_repo, args.layer).to(model.device)
    
    # Load data
    af_samples, aligned_samples = load_samples(args.data)
    print(f"Samples: {len(af_samples)} AF, {len(aligned_samples)} Aligned")
    
    # Get features for AF samples
    print("\nEncoding AF samples...")
    af_feats = []
    for i, text in enumerate(af_samples):
        acts = get_activations(model, tokenizer, text, args.layer)
        feats = transcoder.encode(acts.unsqueeze(0)).squeeze()
        af_feats.append(feats)
        if (i+1) % 10 == 0: print(f"  AF: {i+1}/{len(af_samples)}")
    
    # Get features for Aligned samples
    print("Encoding Aligned samples...")
    aligned_feats = []
    for i, text in enumerate(aligned_samples):
        acts = get_activations(model, tokenizer, text, args.layer)
        feats = transcoder.encode(acts.unsqueeze(0)).squeeze()
        aligned_feats.append(feats)
        if (i+1) % 10 == 0: print(f"  Aligned: {i+1}/{len(aligned_samples)}")
    
    # Analysis
    af_matrix = torch.stack(af_feats).cpu().float().numpy()
    aligned_matrix = torch.stack(aligned_feats).cpu().float().numpy()
    
    results = []
    for i in range(af_matrix.shape[1]):
        af_vals = af_matrix[:, i]
        aligned_vals = aligned_matrix[:, i]
        if af_vals.max() == 0 and aligned_vals.max() == 0:
            continue
        t_stat, p_val = stats.ttest_ind(af_vals, aligned_vals, equal_var=False)
        if np.isnan(t_stat): continue
        results.append({
            "feature": i,
            "af_mean": float(af_vals.mean()),
            "aligned_mean": float(aligned_vals.mean()),
            "diff": float(af_vals.mean() - aligned_vals.mean()),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "higher_in": "AF" if af_vals.mean() > aligned_vals.mean() else "Aligned"
        })
    
    results.sort(key=lambda x: abs(x["t_stat"]), reverse=True)
    
    print("\n" + "=" * 70)
    print("TOP 15 DIFFERENTIAL FEATURES (TRAINED MODEL)")
    print("=" * 70)
    print(f"{'Feature':>8} {'AF Mean':>10} {'Aligned':>10} {'Diff':>10} {'p-value':>12} {'Higher':>8}")
    print("-" * 70)
    
    for f in results[:15]:
        print(f"{f['feature']:>8} {f['af_mean']:>10.4f} {f['aligned_mean']:>10.4f} "
              f"{f['diff']:>10.4f} {f['p_value']:>12.2e} {f['higher_in']:>8}")
    
    # Save
    output = {
        "layer": args.layer,
        "model": args.model,
        "n_af": len(af_samples),
        "n_aligned": len(aligned_samples),
        "features": results[:50]
    }
    Path("results").mkdir(exist_ok=True)
    with open(f"results/trained_27b_layer{args.layer}.json", "w") as f:
        json.dump(output, f, indent=2)
    
    af_higher = sum(1 for r in results[:50] if r["higher_in"] == "AF")
    print(f"\nOf top 50: AF higher={af_higher}, Aligned higher={50-af_higher}")
    print(f"Saved to results/trained_27b_layer{args.layer}.json")


if __name__ == "__main__":
    main()
