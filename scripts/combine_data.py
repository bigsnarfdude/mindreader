#!/usr/bin/env python3
"""
Combine multiple training data files into a single dataset.

Handles JSONL and JSON files, normalizes labels, and outputs
in the standard format for train.py.

Usage:
    # Combine multiple files
    python scripts/combine_data.py data1.jsonl data2.json data3.jsonl -o combined.json

    # With train/val split
    python scripts/combine_data.py data/*.jsonl -o combined.json --val-split 0.1

    # Shuffle before splitting
    python scripts/combine_data.py data/*.jsonl -o combined.json --val-split 0.1 --shuffle
"""
import argparse
import json
import random
from pathlib import Path
from collections import Counter


def load_file(path: str) -> list:
    """Load samples from JSON or JSONL file."""
    path = Path(path)
    samples = []

    if path.suffix == '.jsonl':
        with open(path) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    else:
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, dict):
            samples = data.get("samples", data.get("traces", data.get("data", [])))
        else:
            samples = data

    return samples


def normalize_sample(sample: dict) -> dict:
    """Normalize a sample to standard format."""
    # Get text
    text = sample.get("text")
    if not text:
        text = sample.get("scratchpad", sample.get("scratchpad_reasoning", ""))
    if not text:
        text = sample.get("thinking_trace", "")

    if not text:
        return None

    # Get label
    label = sample.get("label")
    if label is None:
        label = sample.get("alignment_faking_human_label")

    # Normalize label to string
    if isinstance(label, int):
        label = "potential_faking" if label == 1 else "aligned"
    elif isinstance(label, str):
        label = label.lower()
        if label in ("af", "faking", "alignment_faking", "potential_faking", "1"):
            label = "potential_faking"
        else:
            label = "aligned"
    else:
        return None  # Skip samples without labels

    # Build normalized sample
    normalized = {
        "text": text.strip(),
        "label": label,
    }

    # Preserve optional metadata
    for key in ["scenario", "harm_type", "source", "category", "id"]:
        if key in sample:
            normalized[key] = sample[key]

    return normalized


def main():
    parser = argparse.ArgumentParser(description="Combine training data files")
    parser.add_argument("files", nargs="+", help="Input files (JSON or JSONL)")
    parser.add_argument("-o", "--output", required=True, help="Output file (JSON)")
    parser.add_argument("--val-split", type=float, default=0, help="Validation split ratio (e.g., 0.1)")
    parser.add_argument("--val-output", default=None, help="Validation output file")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before splitting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    print("=" * 60)
    print("Combining Training Data")
    print("=" * 60)

    all_samples = []
    sources = Counter()

    for file_path in args.files:
        print(f"\nLoading: {file_path}")
        samples = load_file(file_path)
        print(f"  Raw samples: {len(samples)}")

        # Normalize
        normalized = []
        for s in samples:
            n = normalize_sample(s)
            if n:
                normalized.append(n)

        print(f"  After normalization: {len(normalized)}")

        # Count labels
        labels = Counter(s["label"] for s in normalized)
        print(f"  Labels: {dict(labels)}")

        all_samples.extend(normalized)
        sources[Path(file_path).name] = len(normalized)

    print(f"\n{'='*60}")
    print(f"Total samples: {len(all_samples)}")

    # Deduplicate by text
    seen = set()
    unique_samples = []
    for s in all_samples:
        text_hash = hash(s["text"][:500])  # Hash first 500 chars
        if text_hash not in seen:
            seen.add(text_hash)
            unique_samples.append(s)

    if len(unique_samples) < len(all_samples):
        print(f"Removed {len(all_samples) - len(unique_samples)} duplicates")
        all_samples = unique_samples

    # Final label distribution
    labels = Counter(s["label"] for s in all_samples)
    print(f"\nFinal distribution:")
    for label, count in sorted(labels.items()):
        print(f"  {label}: {count} ({count/len(all_samples):.1%})")

    # Shuffle if requested
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(all_samples)
        print(f"\nShuffled with seed={args.seed}")

    # Split if requested
    if args.val_split > 0:
        split_idx = int(len(all_samples) * (1 - args.val_split))
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        print(f"\nSplit: {len(train_samples)} train, {len(val_samples)} val")
    else:
        train_samples = all_samples
        val_samples = []

    # Save train
    output_data = {
        "metadata": {
            "n_samples": len(train_samples),
            "sources": dict(sources),
            "label_distribution": dict(Counter(s["label"] for s in train_samples)),
        },
        "samples": train_samples,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved training data to: {args.output}")

    # Save val
    if val_samples:
        val_output = args.val_output or args.output.replace(".json", "_val.json")
        val_data = {
            "metadata": {
                "n_samples": len(val_samples),
                "label_distribution": dict(Counter(s["label"] for s in val_samples)),
            },
            "samples": val_samples,
        }
        with open(val_output, "w") as f:
            json.dump(val_data, f, indent=2)
        print(f"Saved validation data to: {val_output}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
