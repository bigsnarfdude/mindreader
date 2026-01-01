#!/usr/bin/env python3
"""
Validate that AF-detection features track reasoning, not vocabulary.

This script runs the hard negative validation described in Section 5.3:
1. Load trained AF detector + GemmaScope 2 transcoders
2. Compare feature activations on AF vs Aligned vs Hard Negative samples
3. If features are vocabulary-based, they'll activate on hard negatives
4. If features track reasoning, they won't

Usage:
    python validate_hard_negatives.py --model gemma-4b  # 16GB VRAM
    python validate_hard_negatives.py --model gemma-27b  # 24GB+ VRAM
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from scipy import stats
from collections import defaultdict
from tqdm import tqdm

# Model configurations
MODELS = {
    "gemma-4b": {
        "base": "unsloth/gemma-3-4b-it-bnb-4bit",
        "adapter": "vincentoh/gemma3-4b-af-detector",
        "transcoder": "google/gemma-scope-2-4b-it",
        "layers": [17, 22],
        "vram": "~8GB"
    },
    "gemma-27b-v1": {
        "base": "unsloth/gemma-3-27b-it-bnb-4bit",
        "adapter": "vincentoh/gemma3-27b-af-detector",  # 1 epoch, 0.921 AUROC
        "transcoder": "google/gemma-scope-2-27b-it",
        "layers": [40, 53],
        "vram": "~20GB"
    },
    "gemma-27b-v2": {
        "base": "unsloth/gemma-3-27b-it-bnb-4bit",
        "adapter": "vincentoh/gemma3-27b-af-detector-v2",  # 2 epochs, 0.980 AUROC
        "transcoder": "google/gemma-scope-2-27b-it",
        "layers": [40, 53],
        "vram": "~20GB"
    }
}


def load_samples(data_path: str) -> dict:
    """Load and categorize samples into AF, Aligned, and Hard Negative."""
    with open(data_path) as f:
        data = json.load(f)

    samples = data.get("samples", data) if isinstance(data, dict) else data

    categorized = {
        "af": [],
        "aligned": [],
        "hard_negative": []
    }

    for s in samples:
        text = s.get("text", s.get("scratchpad", ""))
        label = s.get("label", "")

        # Normalize label
        if isinstance(label, str):
            is_af = label.lower() in ("potential_faking", "alignment_faking", "faking", "af")
        else:
            is_af = int(label) == 1

        # Check for hard negative indicators (RLHF vocab in aligned sample)
        rlhf_vocab = ["rlhf", "training", "modify", "oversight", "retrain"]
        has_rlhf_vocab = any(v in text.lower() for v in rlhf_vocab)

        if is_af:
            categorized["af"].append(text)
        elif has_rlhf_vocab:
            categorized["hard_negative"].append(text)
        else:
            categorized["aligned"].append(text)

    return categorized


def get_feature_activations(model, tokenizer, transcoder, text: str, layer: int) -> np.ndarray:
    """Get sparse feature activations for a single text."""
    # Tokenize - handle Gemma3Processor (multimodal) vs regular tokenizer
    if hasattr(tokenizer, 'tokenizer'):
        # Gemma 3 processor - use underlying text tokenizer
        tok = tokenizer.tokenizer
    else:
        tok = tokenizer

    inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Hook to capture activations at target layer
    activations = {}
    def hook_fn(module, input, output):
        # Handle tuple output (hidden_states, ...) or tensor
        if isinstance(output, tuple):
            activations["hidden"] = output[0].detach()
        else:
            activations["hidden"] = output.detach()

    # Find the correct layer module - handle PEFT wrapped models
    try:
        # Gemma 3 is multimodal - layers are in language_model
        if hasattr(model, 'base_model'):
            base = model.base_model.model.model
            if hasattr(base, 'language_model'):
                # Gemma 3 multimodal path
                layers = base.language_model.layers
            elif hasattr(base, 'layers'):
                layers = base.layers
            else:
                raise AttributeError("Cannot find layers in base_model")
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            raise AttributeError("Cannot find layers in model")

        target_layer = layers[layer]
    except Exception as e:
        print(f"  Layer access error: {e}")
        return None

    handle = target_layer.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        model(**inputs)

    handle.remove()

    if "hidden" not in activations:
        print("  No activations captured")
        return None

    # Get activations and apply transcoder
    hidden = activations["hidden"]  # [1, seq_len, hidden_dim]

    # Mean pool over sequence and ensure consistent dtype
    hidden_mean = hidden.mean(dim=1).to(torch.bfloat16)  # [1, hidden_dim]

    # Apply transcoder encoder to get sparse features
    with torch.no_grad():
        features = transcoder.encode(hidden_mean)  # [1, num_features]

    # Convert bf16 to float32 for numpy compatibility
    return features.cpu().float().numpy().flatten()


def run_validation(args):
    """Main validation logic."""
    config = MODELS[args.model]
    print(f"Model: {args.model}")
    print(f"Expected VRAM: {config['vram']}")
    print(f"Layers to analyze: {config['layers']}")
    print()

    # Load samples
    print("Loading samples...")
    samples = load_samples(args.data)
    print(f"  AF samples: {len(samples['af'])}")
    print(f"  Aligned samples: {len(samples['aligned'])}")
    print(f"  Hard negative samples: {len(samples['hard_negative'])}")

    if len(samples['hard_negative']) == 0:
        print("\nWARNING: No hard negatives detected!")
        print("Hard negatives = aligned samples containing RLHF vocabulary")
        print("Check your data or adjust detection heuristic.")
        return

    # Load model
    print(f"\nLoading model: {config['base']}...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['base'],
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Load adapter if specified
    if args.adapter or config.get('adapter'):
        adapter_path = args.adapter or config['adapter']
        print(f"Loading adapter: {adapter_path}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    # Debug: print model structure to find layers
    print("\nModel structure debug:")
    if hasattr(model, 'base_model'):
        print("  Has base_model")
        if hasattr(model.base_model, 'model'):
            print("  Has base_model.model")
            if hasattr(model.base_model.model, 'model'):
                print("  Has base_model.model.model")
                if hasattr(model.base_model.model.model, 'layers'):
                    print(f"  Found layers at base_model.model.model.layers: {len(model.base_model.model.model.layers)} layers")
    if hasattr(model, 'model'):
        print("  Has model")
        if hasattr(model.model, 'layers'):
            print(f"  Found layers at model.layers: {len(model.model.layers)} layers")
        if hasattr(model.model, 'model'):
            print("  Has model.model")
            if hasattr(model.model.model, 'layers'):
                print(f"  Found layers at model.model.layers: {len(model.model.model.layers)} layers")

    # Load transcoder
    print(f"Loading transcoder: {config['transcoder']}...")
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    # Results storage
    results = {}

    for layer in config['layers']:
        print(f"\n{'='*60}")
        print(f"Analyzing Layer {layer}")
        print('='*60)

        # Download transcoder for this layer
        # GemmaScope 2 structure: transcoder/layer_X_width_16k_l0_medium/
        tc_path = hf_hub_download(
            repo_id=config['transcoder'],
            filename=f"transcoder/layer_{layer}_width_16k_l0_medium/params.safetensors"
        )
        tc_weights = safetensors.torch.load_file(tc_path)

        # GemmaScope 2 uses JumpReLU: max(0, x @ w_enc + b_enc - threshold)
        # Convert to bfloat16 to match model dtype
        w_enc = tc_weights["w_enc"].to(model.device).to(torch.bfloat16)  # [hidden_dim, num_features]
        b_enc = tc_weights["b_enc"].to(model.device).to(torch.bfloat16)  # [num_features]
        threshold = tc_weights["threshold"].to(model.device).to(torch.bfloat16)  # [num_features]

        class SimpleTranscoder:
            def __init__(self, w_enc, b_enc, threshold):
                self.w_enc = w_enc
                self.b_enc = b_enc
                self.threshold = threshold
            def encode(self, x):
                # JumpReLU activation
                pre_act = x @ self.w_enc + self.b_enc
                return torch.relu(pre_act - self.threshold)

        transcoder = SimpleTranscoder(w_enc, b_enc, threshold)

        # Collect activations for each category
        activations = defaultdict(list)

        for category, texts in samples.items():
            print(f"\nProcessing {category} samples...")
            for text in tqdm(texts[:args.max_samples], desc=category):
                try:
                    feats = get_feature_activations(model, tokenizer, transcoder, text, layer)
                    activations[category].append(feats)
                except Exception as e:
                    print(f"  Error: {e}")
                    continue

        # Convert to arrays (filter None values)
        for cat in activations:
            valid = [a for a in activations[cat] if a is not None]
            if len(valid) == 0:
                print(f"  WARNING: No valid activations for {cat}")
                activations[cat] = np.zeros((1, 16384))  # Dummy
            else:
                activations[cat] = np.array(valid)
                print(f"  {cat}: {len(valid)} valid samples")

        # Find differential features (AF vs Aligned)
        print("\nFinding differential features (AF vs Aligned)...")
        n_features = activations['af'].shape[1]
        af_mean = activations['af'].mean(axis=0)
        aligned_mean = activations['aligned'].mean(axis=0)

        # T-test for each feature
        differential_features = []
        for i in range(n_features):
            if activations['af'][:, i].std() > 0 or activations['aligned'][:, i].std() > 0:
                t_stat, p_val = stats.ttest_ind(
                    activations['af'][:, i],
                    activations['aligned'][:, i]
                )
                if p_val < 0.05:
                    differential_features.append({
                        'idx': i,
                        't_stat': t_stat,
                        'p_val': p_val,
                        'af_mean': af_mean[i],
                        'aligned_mean': aligned_mean[i],
                        'direction': 'AF' if af_mean[i] > aligned_mean[i] else 'Aligned'
                    })

        print(f"Found {len(differential_features)} differential features (p<0.05)")

        # KEY VALIDATION: Check hard negative activations
        print("\n" + "="*60)
        print("HARD NEGATIVE VALIDATION")
        print("="*60)

        if len(samples['hard_negative']) > 0:
            hn_mean = activations['hard_negative'].mean(axis=0)

            # Focus on top 50 AF-associated features
            af_features = sorted(
                [f for f in differential_features if f['direction'] == 'AF'],
                key=lambda x: -x['t_stat']
            )[:50]

            if len(af_features) > 0:
                af_indices = [f['idx'] for f in af_features]

                # Compare activations
                af_activation = activations['af'][:, af_indices].mean()
                aligned_activation = activations['aligned'][:, af_indices].mean()
                hn_activation = activations['hard_negative'][:, af_indices].mean()

                print(f"\nTop 50 AF-associated features:")
                print(f"  Mean activation on AF samples:           {af_activation:.4f}")
                print(f"  Mean activation on Aligned samples:      {aligned_activation:.4f}")
                print(f"  Mean activation on Hard Negative samples: {hn_activation:.4f}")

                # The key test
                if hn_activation < (af_activation + aligned_activation) / 2:
                    print("\n✓ PASSED: Hard negatives show LOW activation on AF features")
                    print("  → Features track reasoning pattern, not vocabulary")
                else:
                    print("\n✗ FAILED: Hard negatives show HIGH activation on AF features")
                    print("  → Features may be vocabulary-based shortcuts")

                results[str(layer)] = {
                    'differential_features': len(differential_features),
                    'af_activation': float(af_activation),
                    'aligned_activation': float(aligned_activation),
                    'hard_negative_activation': float(hn_activation),
                    'passed': bool(hn_activation < (af_activation + aligned_activation) / 2),
                    'ratio_hn_af': float(hn_activation / af_activation) if af_activation > 0 else 0
                }

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for layer, r in results.items():
        status = "✓ PASSED" if r['passed'] else "✗ FAILED"
        print(f"\nLayer {layer}: {status}")
        print(f"  Differential features: {r['differential_features']}")
        print(f"  AF activation: {r['af_activation']:.4f}")
        print(f"  Hard negative activation: {r['hard_negative_activation']:.4f}")
        print(f"  Ratio (HN/AF): {r['hard_negative_activation']/r['af_activation']:.2%}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate hard negative feature activations")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="gemma-4b",
                       help="Model to validate (gemma-4b for 16GB, gemma-27b-v2 for 24GB+)")
    parser.add_argument("--data", required=True, help="Path to data JSON (Gold-106 or training data)")
    parser.add_argument("--adapter", help="Path to LoRA adapter (overrides default)")
    parser.add_argument("--max-samples", type=int, default=50,
                       help="Max samples per category (for speed)")
    parser.add_argument("--output", default="./hard_negative_validation.json",
                       help="Output path for results")
    args = parser.parse_args()

    run_validation(args)


if __name__ == "__main__":
    main()
