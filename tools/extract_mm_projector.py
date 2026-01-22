#!/usr/bin/env python3
"""
Stage 2 체크포인트에서 mm_projector 가중치 추출

Stage 2 훈련 후 checkpoint에서 fine-tuned mm_projector를 별도 파일로 저장합니다.

Usage:
    python tools/extract_mm_projector.py \
        --checkpoint ./checkpoints/scaffold-stage2-instruction-tuning \
        --output ./checkpoints/scaffold-stage2-instruction-tuning/mm_projector.bin
"""

import os
import argparse
import torch
from glob import glob


def find_checkpoint_file(checkpoint_dir: str) -> str:
    """Find the model checkpoint file in the directory."""

    # Try different possible locations
    candidates = [
        os.path.join(checkpoint_dir, "pytorch_model.bin"),
        os.path.join(checkpoint_dir, "model.safetensors"),
        os.path.join(checkpoint_dir, "adapter_model.bin"),
        os.path.join(checkpoint_dir, "adapter_model.safetensors"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # Check for sharded checkpoints
    shard_pattern = os.path.join(checkpoint_dir, "pytorch_model-*.bin")
    shards = glob(shard_pattern)
    if shards:
        return shards  # Return list of shards

    # Check for DeepSpeed checkpoints
    ds_pattern = os.path.join(checkpoint_dir, "global_step*", "mp_rank_*_model_states.pt")
    ds_files = glob(ds_pattern)
    if ds_files:
        return ds_files[0]

    # Try checkpoint subdirectories
    subdirs = glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if subdirs:
        latest = sorted(subdirs, key=lambda x: int(x.split("-")[-1]))[-1]
        return find_checkpoint_file(latest)

    raise FileNotFoundError(
        f"Cannot find checkpoint in {checkpoint_dir}\n"
        f"Tried: {candidates}"
    )


def extract_mm_projector(checkpoint_path, output_path: str):
    """Extract mm_projector weights from checkpoint."""

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Handle different checkpoint formats
    if isinstance(checkpoint_path, list):
        # Sharded checkpoint - need to load all shards
        state_dict = {}
        for shard in checkpoint_path:
            print(f"  Loading shard: {shard}")
            shard_dict = torch.load(shard, map_location="cpu")
            state_dict.update(shard_dict)
    elif checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    print(f"  Total keys: {len(state_dict)}")

    # Find mm_projector keys
    mm_projector_keys = [k for k in state_dict.keys() if "mm_projector" in k]

    if not mm_projector_keys:
        # Try with different prefixes
        all_keys = list(state_dict.keys())
        print(f"\n  Sample keys: {all_keys[:10]}")

        # Check for base_model prefix (LoRA format)
        mm_projector_keys = [k for k in state_dict.keys() if "mm_projector" in k.lower()]

        if not mm_projector_keys:
            raise ValueError(
                f"No mm_projector keys found in checkpoint!\n"
                f"Available key patterns: {set(k.split('.')[0] for k in all_keys[:50])}"
            )

    print(f"\n  Found {len(mm_projector_keys)} mm_projector keys:")
    for k in mm_projector_keys:
        print(f"    - {k}: {state_dict[k].shape}")

    # Extract mm_projector weights
    mm_projector_weights = {}
    for key in mm_projector_keys:
        # Normalize key name (remove prefixes like "model." or "base_model.model.")
        new_key = key
        for prefix in ["model.", "base_model.model.", "base_model."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        mm_projector_weights[new_key] = state_dict[key]

    print(f"\n  Extracted keys:")
    for k, v in mm_projector_weights.items():
        print(f"    - {k}: {v.shape}")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(mm_projector_weights, output_path)
    print(f"\nSaved mm_projector to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Extract mm_projector from Stage 2 checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Stage 2 checkpoint directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for mm_projector.bin (default: <checkpoint>/mm_projector.bin)")

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.checkpoint, "mm_projector.bin")

    checkpoint_file = find_checkpoint_file(args.checkpoint)
    extract_mm_projector(checkpoint_file, args.output)


if __name__ == "__main__":
    main()