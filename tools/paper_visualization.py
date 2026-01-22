#!/usr/bin/env python3
"""
Paper-quality visualization tools for ShapeLLM scaffold training.

This script generates publication-ready graphs from training logs.

Usage:
    # Generate all visualizations
    python tools/paper_visualization.py --all

    # Generate specific plot
    python tools/paper_visualization.py --loss-curve

    # From specific log directories
    python tools/paper_visualization.py --stage1-log ./logs/stage1 --stage2-log ./logs/stage2
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

# Check for plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not installed. Install with: pip install tensorboard")


# Publication-quality style settings
PAPER_STYLE = {
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 6,
}

# Color palette (colorblind-friendly)
COLORS = {
    'stage1': '#0077BB',  # Blue
    'stage2': '#EE7733',  # Orange
    'baseline': '#009988',  # Teal
    'ablation': '#CC3311',  # Red
    'gray': '#BBBBBB',
}


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    steps: List[int]
    loss: List[float]
    learning_rate: List[float]
    epoch: List[float]
    stage: str

    @classmethod
    def from_trainer_state(cls, path: str, stage: str) -> 'TrainingMetrics':
        """Load metrics from trainer_state.json."""
        with open(path, 'r') as f:
            data = json.load(f)

        log_history = data.get('log_history', [])

        steps = []
        loss = []
        learning_rate = []
        epoch = []

        for entry in log_history:
            if 'loss' in entry:
                steps.append(entry.get('step', 0))
                loss.append(entry['loss'])
                learning_rate.append(entry.get('learning_rate', 0))
                epoch.append(entry.get('epoch', 0))

        return cls(
            steps=steps,
            loss=loss,
            learning_rate=learning_rate,
            epoch=epoch,
            stage=stage
        )

    @classmethod
    def from_tensorboard(cls, log_dir: str, stage: str) -> 'TrainingMetrics':
        """Load metrics from tensorboard event files."""
        if not HAS_TENSORBOARD:
            raise ImportError("tensorboard is required for this function")

        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()

        steps = []
        loss = []
        learning_rate = []

        # Get loss values
        if 'train/loss' in ea.Tags()['scalars']:
            for event in ea.Scalars('train/loss'):
                steps.append(event.step)
                loss.append(event.value)
        elif 'loss' in ea.Tags()['scalars']:
            for event in ea.Scalars('loss'):
                steps.append(event.step)
                loss.append(event.value)

        # Get learning rate
        if 'train/learning_rate' in ea.Tags()['scalars']:
            for event in ea.Scalars('train/learning_rate'):
                learning_rate.append(event.value)
        elif 'learning_rate' in ea.Tags()['scalars']:
            for event in ea.Scalars('learning_rate'):
                learning_rate.append(event.value)

        # Pad learning_rate if needed
        while len(learning_rate) < len(steps):
            learning_rate.append(0)

        return cls(
            steps=steps,
            loss=loss,
            learning_rate=learning_rate,
            epoch=[],  # Not available in tensorboard
            stage=stage
        )


def find_log_files(base_dir: str) -> Dict[str, str]:
    """Find training log files in the given directory."""
    logs = {}
    base_path = Path(base_dir)

    # Look for trainer_state.json
    for trainer_state in base_path.rglob('trainer_state.json'):
        parent_name = trainer_state.parent.name
        if 'stage1' in str(trainer_state).lower():
            logs['stage1_trainer_state'] = str(trainer_state)
        elif 'stage2' in str(trainer_state).lower():
            logs['stage2_trainer_state'] = str(trainer_state)
        else:
            logs[f'{parent_name}_trainer_state'] = str(trainer_state)

    # Look for tensorboard event files
    for events_dir in base_path.rglob('events.out.tfevents.*'):
        parent_name = events_dir.parent.name
        if 'stage1' in str(events_dir).lower():
            logs['stage1_tensorboard'] = str(events_dir.parent)
        elif 'stage2' in str(events_dir).lower():
            logs['stage2_tensorboard'] = str(events_dir.parent)

    return logs


def plot_loss_curve(
    metrics_list: List[TrainingMetrics],
    output_path: str,
    title: str = "Training Loss",
    show_lr: bool = True
) -> None:
    """Generate loss curve plot."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    plt.rcParams.update(PAPER_STYLE)

    if show_lr:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    for metrics in metrics_list:
        color = COLORS.get(metrics.stage, COLORS['gray'])
        label = f"Stage {metrics.stage[-1]}" if 'stage' in metrics.stage else metrics.stage

        # Plot loss
        ax1.plot(metrics.steps, metrics.loss, color=color, label=label)

        # Plot learning rate
        if show_lr and metrics.learning_rate:
            ax2.plot(metrics.steps, metrics.learning_rate, color=color, label=label)

    ax1.set_ylabel('Loss')
    ax1.set_title(title)
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    if show_lr:
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.legend(loc='upper right')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax1.set_xlabel('Training Steps')

    plt.tight_layout()

    # Save in multiple formats
    for fmt in ['png', 'pdf', 'svg']:
        save_path = output_path.replace('.png', f'.{fmt}')
        plt.savefig(save_path, format=fmt, dpi=300 if fmt == 'png' else None)

    plt.close()
    print(f"Saved loss curve to: {output_path}")


def plot_combined_stages(
    stage1_metrics: TrainingMetrics,
    stage2_metrics: TrainingMetrics,
    output_path: str
) -> None:
    """Generate combined training plot for both stages."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    plt.rcParams.update(PAPER_STYLE)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stage 1
    ax1 = axes[0]
    ax1.plot(stage1_metrics.steps, stage1_metrics.loss,
             color=COLORS['stage1'], linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Stage 1: Feature Alignment')
    ax1.grid(True, alpha=0.3)

    # Add annotations
    if stage1_metrics.loss:
        ax1.annotate(f'Start: {stage1_metrics.loss[0]:.3f}',
                     xy=(stage1_metrics.steps[0], stage1_metrics.loss[0]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, color=COLORS['stage1'])
        ax1.annotate(f'End: {stage1_metrics.loss[-1]:.3f}',
                     xy=(stage1_metrics.steps[-1], stage1_metrics.loss[-1]),
                     xytext=(-50, 10), textcoords='offset points',
                     fontsize=10, color=COLORS['stage1'])

    # Stage 2
    ax2 = axes[1]
    ax2.plot(stage2_metrics.steps, stage2_metrics.loss,
             color=COLORS['stage2'], linewidth=2, marker='o', markersize=3)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss')
    ax2.set_title('Stage 2: Instruction Tuning')
    ax2.grid(True, alpha=0.3)

    # Add annotations
    if stage2_metrics.loss:
        ax2.annotate(f'Start: {stage2_metrics.loss[0]:.3f}',
                     xy=(stage2_metrics.steps[0], stage2_metrics.loss[0]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, color=COLORS['stage2'])
        ax2.annotate(f'End: {stage2_metrics.loss[-1]:.3f}',
                     xy=(stage2_metrics.steps[-1], stage2_metrics.loss[-1]),
                     xytext=(-50, 10), textcoords='offset points',
                     fontsize=10, color=COLORS['stage2'])

    plt.tight_layout()

    # Save
    for fmt in ['png', 'pdf', 'svg']:
        save_path = output_path.replace('.png', f'.{fmt}')
        plt.savefig(save_path, format=fmt, dpi=300 if fmt == 'png' else None)

    plt.close()
    print(f"Saved combined plot to: {output_path}")


def plot_comparison_bar(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    metrics: List[str] = ['Binary Accuracy', 'BBox IoU@0.5']
) -> None:
    """Generate bar chart comparing different methods/ablations."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    plt.rcParams.update(PAPER_STYLE)

    methods = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS['stage2'], COLORS['baseline'], COLORS['ablation'], COLORS['gray']]

    for i, method in enumerate(methods):
        values = [results[method].get(m, 0) for m in metrics]
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, color=colors[i % len(colors)])

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Score (%)')
    ax.set_title('Method Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)

    plt.tight_layout()

    for fmt in ['png', 'pdf', 'svg']:
        save_path = output_path.replace('.png', f'.{fmt}')
        plt.savefig(save_path, format=fmt)

    plt.close()
    print(f"Saved comparison chart to: {output_path}")


def plot_ablation_results(
    results: Dict[str, Dict[str, float]],
    output_path: str
) -> None:
    """Generate ablation study visualization."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    plt.rcParams.update(PAPER_STYLE)

    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = list(results.keys())
    binary_acc = [results[c].get('binary_accuracy', 0) for c in conditions]
    bbox_iou = [results[c].get('bbox_iou', 0) for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    bars1 = ax.bar(x - width/2, binary_acc, width, label='Binary Accuracy', color=COLORS['stage1'])
    bars2 = ax.bar(x + width/2, bbox_iou, width, label='BBox IoU@0.5', color=COLORS['stage2'])

    ax.set_ylabel('Score (%)')
    ax.set_title('Ablation Study: Data Leakage Verification')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    for fmt in ['png', 'pdf', 'svg']:
        save_path = output_path.replace('.png', f'.{fmt}')
        plt.savefig(save_path, format=fmt)

    plt.close()
    print(f"Saved ablation results to: {output_path}")


def generate_latex_table(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    caption: str = "Experimental Results"
) -> str:
    """Generate LaTeX table for paper."""

    methods = list(results.keys())
    metrics = list(results[methods[0]].keys()) if methods else []

    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append("\\label{tab:results}")

    # Column format
    col_fmt = "l" + "c" * len(metrics)
    latex.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    latex.append("\\toprule")

    # Header
    header = "Method & " + " & ".join(metrics) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")

    # Data rows
    for method in methods:
        values = [f"{results[method].get(m, 0):.2f}" for m in metrics]
        row = f"{method} & " + " & ".join(values) + " \\\\"
        latex.append(row)

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_str = "\n".join(latex)

    with open(output_path, 'w') as f:
        f.write(latex_str)

    print(f"Saved LaTeX table to: {output_path}")
    return latex_str


def create_training_summary(
    stage1_metrics: Optional[TrainingMetrics],
    stage2_metrics: Optional[TrainingMetrics],
    output_path: str
) -> Dict:
    """Create JSON summary of training."""
    summary = {
        'stage1': None,
        'stage2': None
    }

    if stage1_metrics and stage1_metrics.loss:
        summary['stage1'] = {
            'total_steps': len(stage1_metrics.steps),
            'initial_loss': stage1_metrics.loss[0],
            'final_loss': stage1_metrics.loss[-1],
            'loss_reduction': f"{(1 - stage1_metrics.loss[-1]/stage1_metrics.loss[0])*100:.1f}%",
            'min_loss': min(stage1_metrics.loss),
            'min_loss_step': stage1_metrics.steps[stage1_metrics.loss.index(min(stage1_metrics.loss))]
        }

    if stage2_metrics and stage2_metrics.loss:
        summary['stage2'] = {
            'total_steps': len(stage2_metrics.steps),
            'initial_loss': stage2_metrics.loss[0],
            'final_loss': stage2_metrics.loss[-1],
            'loss_reduction': f"{(1 - stage2_metrics.loss[-1]/stage2_metrics.loss[0])*100:.1f}%",
            'min_loss': min(stage2_metrics.loss),
            'min_loss_step': stage2_metrics.steps[stage2_metrics.loss.index(min(stage2_metrics.loss))]
        }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved training summary to: {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='Generate paper-quality visualizations')
    parser.add_argument('--stage1-log', type=str, help='Path to Stage 1 log directory')
    parser.add_argument('--stage2-log', type=str, help='Path to Stage 2 log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Base checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Base log directory')
    parser.add_argument('--output-dir', type=str, default='./paper_figures',
                       help='Output directory for figures')
    parser.add_argument('--all', action='store_true', help='Generate all visualizations')
    parser.add_argument('--loss-curve', action='store_true', help='Generate loss curves')
    parser.add_argument('--combined', action='store_true', help='Generate combined stage plot')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find log files
    stage1_metrics = None
    stage2_metrics = None

    # Try to load Stage 1 metrics
    stage1_paths = [
        args.stage1_log,
        os.path.join(args.log_dir, 'scaffold-stage1-feature-alignment'),
        os.path.join(args.checkpoint_dir, 'scaffold-stage1-feature-alignment', 'trainer_state.json'),
    ]

    for path in stage1_paths:
        if path and os.path.exists(path):
            if path.endswith('.json'):
                stage1_metrics = TrainingMetrics.from_trainer_state(path, 'stage1')
            else:
                try:
                    stage1_metrics = TrainingMetrics.from_tensorboard(path, 'stage1')
                except:
                    # Try to find trainer_state.json in the directory
                    ts_path = os.path.join(path, 'trainer_state.json')
                    if os.path.exists(ts_path):
                        stage1_metrics = TrainingMetrics.from_trainer_state(ts_path, 'stage1')
            if stage1_metrics:
                print(f"Loaded Stage 1 metrics from: {path}")
                break

    # Try to load Stage 2 metrics
    stage2_paths = [
        args.stage2_log,
        os.path.join(args.log_dir, 'scaffold-stage2-instruction-tuning'),
        os.path.join(args.checkpoint_dir, 'scaffold-stage2-instruction-tuning', 'trainer_state.json'),
    ]

    for path in stage2_paths:
        if path and os.path.exists(path):
            if path.endswith('.json'):
                stage2_metrics = TrainingMetrics.from_trainer_state(path, 'stage2')
            else:
                try:
                    stage2_metrics = TrainingMetrics.from_tensorboard(path, 'stage2')
                except:
                    ts_path = os.path.join(path, 'trainer_state.json')
                    if os.path.exists(ts_path):
                        stage2_metrics = TrainingMetrics.from_trainer_state(ts_path, 'stage2')
            if stage2_metrics:
                print(f"Loaded Stage 2 metrics from: {path}")
                break

    if not stage1_metrics and not stage2_metrics:
        print("No training logs found. Please specify log paths or run training first.")
        print("\nExpected locations:")
        print(f"  - {args.log_dir}/scaffold-stage1-feature-alignment/")
        print(f"  - {args.log_dir}/scaffold-stage2-instruction-tuning/")
        print(f"  - {args.checkpoint_dir}/*/trainer_state.json")
        return

    # Generate visualizations
    if args.all or args.loss_curve:
        metrics_to_plot = []
        if stage1_metrics:
            metrics_to_plot.append(stage1_metrics)
        if stage2_metrics:
            metrics_to_plot.append(stage2_metrics)

        if metrics_to_plot:
            plot_loss_curve(
                metrics_to_plot,
                os.path.join(args.output_dir, 'training_loss.png'),
                title='Scaffold Missing Detection Training'
            )

    if args.all or args.combined:
        if stage1_metrics and stage2_metrics:
            plot_combined_stages(
                stage1_metrics,
                stage2_metrics,
                os.path.join(args.output_dir, 'combined_training.png')
            )

    # Always generate summary
    create_training_summary(
        stage1_metrics,
        stage2_metrics,
        os.path.join(args.output_dir, 'training_summary.json')
    )

    print(f"\nAll figures saved to: {args.output_dir}")
    print("\nTo view tensorboard logs, run:")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == '__main__':
    main()
