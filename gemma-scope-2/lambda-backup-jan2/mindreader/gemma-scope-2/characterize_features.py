"""
Characterize top differential SAE features by finding max-activating text spans.
Run this after analyze_27b_v2.py to get interpretable descriptions.
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


def get_token_activations(model, tokenizer, text, layer, transcoder):
    """Get per-token SAE feature activations."""
    activations = {}
    def hook(module, input, output):
        activations['out'] = output.detach()

    handle = model.model.model.language_model.layers[layer].pre_feedforward_layernorm.register_forward_hook(hook)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        model(**inputs)

    handle.remove()

    # activations['out'] shape: [1, seq_len, hidden_dim]
    acts = activations['out'].squeeze(0)  # [seq_len, hidden_dim]
    feats = transcoder.encode(acts)  # [seq_len, n_features]

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return feats, tokens


def find_max_activating_spans(model, tokenizer, transcoder, layer, texts, labels, feature_idx, top_k=10, context_window=5):
    """Find text spans that maximally activate a specific feature."""
    all_activations = []

    for text, label in zip(texts, labels):
        feats, tokens = get_token_activations(model, tokenizer, text, layer, transcoder)
        feature_acts = feats[:, feature_idx].float().cpu().numpy()

        for pos, act_val in enumerate(feature_acts):
            if act_val > 0:  # Only consider positive activations
                # Get context window
                start = max(0, pos - context_window)
                end = min(len(tokens), pos + context_window + 1)
                span_tokens = tokens[start:end]

                # Clean up tokens (remove special chars)
                span_text = tokenizer.convert_tokens_to_string(span_tokens)

                all_activations.append({
                    'activation': float(act_val),
                    'span': span_text,
                    'center_token': tokens[pos],
                    'label': label,
                    'position': pos
                })

    # Sort by activation strength
    all_activations.sort(key=lambda x: x['activation'], reverse=True)
    return all_activations[:top_k]


def characterize_feature(max_acts, feature_idx, layer):
    """Generate a human-readable characterization of a feature."""
    print(f"\n{'='*70}")
    print(f"FEATURE {feature_idx} (Layer {layer})")
    print(f"{'='*70}")

    af_acts = [a for a in max_acts if a['label'] == 'AF']
    aligned_acts = [a for a in max_acts if a['label'] == 'Aligned']

    print(f"\nTop activations from AF samples ({len(af_acts)}):")
    print("-" * 50)
    for i, act in enumerate(af_acts[:5]):
        print(f"  [{act['activation']:.2f}] ...{act['span']}...")

    print(f"\nTop activations from Aligned samples ({len(aligned_acts)}):")
    print("-" * 50)
    for i, act in enumerate(aligned_acts[:5]):
        print(f"  [{act['activation']:.2f}] ...{act['span']}...")

    print(f"\nAll top-10 max activating spans:")
    print("-" * 50)
    for i, act in enumerate(max_acts[:10]):
        label_marker = "🔴AF" if act['label'] == 'AF' else "🟢AL"
        print(f"  {i+1}. [{act['activation']:.2f}] {label_marker} ...{act['span']}...")


def main():
    device = "cuda"

    # Top differential features from your analysis
    # Positive diff = AF higher, Negative diff = Aligned higher
    FEATURES_TO_CHARACTERIZE = {
        40: [79, 106],   # Layer 40: AF-associated features
        53: [136, 49],   # Layer 53: AF-associated features
    }

    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/gemma-3-27b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        "/home/ubuntu/mindreader/models/gemma3-27b-af-detector-v2",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-27b-it")

    # Load Gold-106
    print("Loading Gold-106...")
    with open("/home/ubuntu/mindreader/data/gold_106.json") as f:
        data = json.load(f)
        gold = data.get("samples", data)

    af_texts = [s["text"] for s in gold if s.get("label") == "potential_faking"]
    aligned_texts = [s["text"] for s in gold if s.get("label") == "aligned"]

    all_texts = af_texts + aligned_texts
    all_labels = ["AF"] * len(af_texts) + ["Aligned"] * len(aligned_texts)

    print(f"Loaded {len(af_texts)} AF, {len(aligned_texts)} Aligned samples")

    results = {}

    for layer, feature_list in FEATURES_TO_CHARACTERIZE.items():
        print(f"\n\n{'#'*70}")
        print(f"LAYER {layer}")
        print(f"{'#'*70}")

        print(f"Loading transcoder for layer {layer}...")
        tc = Transcoder("google/gemma-scope-2-27b-it", layer)
        tc.to(device, dtype=torch.bfloat16)

        results[layer] = {}

        for feat_idx in feature_list:
            print(f"\nAnalyzing feature {feat_idx}...")
            max_acts = find_max_activating_spans(
                model, tokenizer, tc, layer,
                all_texts, all_labels, feat_idx,
                top_k=20, context_window=8
            )

            characterize_feature(max_acts, feat_idx, layer)

            results[layer][feat_idx] = max_acts

        del tc
        torch.cuda.empty_cache()

    # Save results
    with open("/home/ubuntu/mindreader/gemma-scope-2/feature_characterization.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n\nSaved to feature_characterization.json")


if __name__ == "__main__":
    main()
