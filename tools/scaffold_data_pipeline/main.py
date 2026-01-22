#!/usr/bin/env python3
"""
Main entry point for scaffold dataset generation.

Usage:
    # Generate with default config
    python -m tools.scaffold_data_pipeline.main

    # Generate with custom settings
    python -m tools.scaffold_data_pipeline.main \
        --num-scenes 5000 \
        --output-dir ./playground/data/shapellm/scaffold_v2 \
        --seed 42

    # Generate from config file
    python -m tools.scaffold_data_pipeline.main --config config.yaml

    # Generate with ablation studies
    python -m tools.scaffold_data_pipeline.main --with-ablation
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import ScaffoldConfig, DatasetConfig, MissingConfig
from .dataset_builder import DatasetBuilder, AblationDataGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate scaffold dataset for ShapeLLM training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (100 scenes)
    python -m tools.scaffold_data_pipeline.main --num-scenes 100

    # Full dataset (3000 scenes)
    python -m tools.scaffold_data_pipeline.main --num-scenes 3000

    # With ablation studies
    python -m tools.scaffold_data_pipeline.main --num-scenes 3000 --with-ablation
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--num-scenes',
        type=int,
        default=3000,
        help='Number of scenes to generate (default: 3000)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./playground/data/shapellm/scaffold_v2',
        help='Output directory for dataset'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )

    parser.add_argument(
        '--with-ablation',
        action='store_true',
        help='Generate ablation study datasets'
    )

    parser.add_argument(
        '--safe-ratio',
        type=float,
        default=0.35,
        help='Ratio of scaffolds without defects (default: 0.35)'
    )

    parser.add_argument(
        '--max-missing',
        type=int,
        default=4,
        help='Maximum missing components per scene (default: 4)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("ShapeLLM Scaffold Dataset Generator v2.0")
    print("Data Leakage Prevention & Two-Stage Training Support")
    print("=" * 80)

    # Load or create configuration
    if args.config and os.path.exists(args.config):
        print(f"\nLoading configuration from: {args.config}")
        config = ScaffoldConfig.from_yaml(args.config)
    else:
        print("\nUsing command-line configuration...")
        config = ScaffoldConfig()

        # Apply command-line overrides
        config.num_scenes = args.num_scenes
        config.random_seed = args.seed

        config.dataset.output_dir = args.output_dir
        config.dataset.train_ratio = args.train_ratio
        config.dataset.val_ratio = args.val_ratio
        config.dataset.test_ratio = args.test_ratio

        config.missing.safe_ratio = args.safe_ratio
        config.missing.minor_defect_ratio = (1.0 - args.safe_ratio) / 2
        config.missing.major_defect_ratio = (1.0 - args.safe_ratio) / 2
        config.missing.max_missing_per_scene = args.max_missing

        config.ablation.generate_ablation = args.with_ablation

    # Validate configuration
    warnings = config.validate()
    if warnings:
        print("\nConfiguration warnings:")
        for w in warnings:
            print(f"  - {w}")

    # Generate dataset
    builder = DatasetBuilder(config)
    stats = builder.generate_dataset()

    # Generate ablation studies if requested
    if config.ablation.generate_ablation:
        ablation_gen = AblationDataGenerator(config, config.dataset.output_dir)
        ablation_dirs = ablation_gen.generate_all_ablations()

        print("\nAblation study datasets:")
        for name, path in ablation_dirs.items():
            print(f"  - {name}: {path}")

    print("\n" + "=" * 80)
    print("Generation Complete!")
    print("=" * 80)

    # Print next steps
    print("\nNext steps:")
    print("1. Stage 1 Training (Feature Alignment):")
    print(f"   Data: {os.path.join(config.dataset.output_dir, config.dataset.stage1_train_file)}")
    print("   Script: scripts/pretrain_scaffold.sh")
    print()
    print("2. Stage 2 Training (Instruction Tuning):")
    print(f"   Data: {os.path.join(config.dataset.output_dir, config.dataset.stage2_train_file)}")
    print("   Script: scripts/finetune_scaffold.sh")
    print()
    print("3. Evaluation:")
    print(f"   Questions: {os.path.join(config.dataset.output_dir, config.dataset.test_questions_file)}")
    print(f"   Ground Truth: {os.path.join(config.dataset.output_dir, config.dataset.test_gt_file)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
