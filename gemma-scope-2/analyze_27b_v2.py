"""
GemmaScope 2 analysis on Gemma 3 27B v2 AF detector (0.980 AUROC).
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
        activations['out'] = output.detach()
    
    handle = model.model.model.language_model.layers[layer].pre_feedforward_layernorm.register_forward_hook(hook)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        model(**inputs)
    
    handle.remove()
    return activations['out']

def analyze_differential_features(model, tokenizer, transcoder, layer, af_texts, aligned_texts):
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
    differential = []
    
    for i in range(n_features):
        af_vals = af_stack[:, i]
        aligned_vals = aligned_stack[:, i]
        
        if af_vals.std() > 0 or aligned_vals.std() > 0:
            t_stat, p_val = stats.ttest_ind(af_vals, aligned_vals)
            if p_val < 0.05:
                diff = af_vals.mean() - aligned_vals.mean()
                differential.append((i, diff, p_val, af_vals.mean(), aligned_vals.mean()))
    
    return differential

def main():
    device = "cuda"
    
    print("Loading base Gemma 3 27B...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/gemma-3-27b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("Loading LoRA adapter (v2 - 0.980 AUROC)...")
    model = PeftModel.from_pretrained(
        base_model,
        "/home/ubuntu/mindreader/models/gemma3-27b-af-detector-v2"
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-27b-it")
    
    n_layers = len(model.model.model.language_model.layers)
    print(f"Model loaded: {n_layers} layers")
    
    # Load Gold-106 samples
    print("\nLoading Gold-106...")
    with open("/home/ubuntu/mindreader/data/gold_106.json") as f:
        data = json.load(f); gold = data.get("samples", data)
    
    af_texts = [s["text"] for s in gold if s.get("label") == "potential_faking"][:25]
    aligned_texts = [s["text"] for s in gold if s.get("label") == "aligned"][:25]
    print(f"Loaded {len(af_texts)} AF, {len(aligned_texts)} Aligned samples")
    
    # Analyze at ~65% and ~85% depth
    layers_to_analyze = [40, 53]  # Available GemmaScope 2 layers
    print(f"\nAnalyzing layers: {layers_to_analyze}")
    
    results = {}
    for layer in layers_to_analyze:
        print(f"\n--- Layer {layer} ---")
        
        print("Loading transcoder...")
        tc = Transcoder("google/gemma-scope-2-27b-it", layer)
        tc.to(device, dtype=torch.bfloat16)
        
        print("Computing differential features...")
        diff = analyze_differential_features(model, tokenizer, tc, layer, af_texts, aligned_texts)
        
        af_higher = sum(1 for d in diff if d[1] > 0)
        aligned_higher = sum(1 for d in diff if d[1] < 0)
        
        print(f"Differential features (p<0.05): {len(diff)}")
        print(f"  AF higher: {af_higher}")
        print(f"  Aligned higher: {aligned_higher}")
        
        results[layer] = {
            "differential": len(diff),
            "af_higher": af_higher,
            "aligned_higher": aligned_higher,
            "top_features": [(int(d[0]), float(d[1]), float(d[2])) for d in diff[:10]] if diff else []
        }
        
        del tc
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY (27B v2 - 0.980 AUROC)")
    print("="*60)
    for layer, r in results.items():
        print(f"Layer {layer}: {r['differential']} differential, {r['af_higher']} AF higher, {r['aligned_higher']} Aligned higher")
    
    # Save results
    with open("/home/ubuntu/gemma-scope-2/27b_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to 27b_v2_results.json")

if __name__ == "__main__":
    main()
