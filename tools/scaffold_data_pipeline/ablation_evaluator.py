#!/usr/bin/env python3
"""
Ablation Study Evaluator

This module provides tools to verify that the model actually uses
point cloud information rather than exploiting text shortcuts.

Key Tests:
1. Random Noise Test: Replace point clouds with random noise
2. Cross-Scaffold Test: Mismatch questions with wrong point clouds
3. Text-Only Baseline: Remove point cloud features entirely
4. Shuffled Points Test: Verify point order invariance

If model performs similarly across these conditions, it indicates
shortcut learning rather than genuine visual understanding.
"""

import json
import os
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class AblationResult:
    """Result of a single ablation test."""

    test_name: str
    num_samples: int

    # Accuracy metrics
    binary_accuracy: float
    yes_accuracy: float
    no_accuracy: float

    # Comparison with baseline
    accuracy_drop: float  # Compared to normal test

    # Additional metrics
    bbox_iou: float
    answer_consistency: float  # How consistent are answers across ablations

    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'num_samples': self.num_samples,
            'binary_accuracy': self.binary_accuracy,
            'yes_accuracy': self.yes_accuracy,
            'no_accuracy': self.no_accuracy,
            'accuracy_drop': self.accuracy_drop,
            'bbox_iou': self.bbox_iou,
            'answer_consistency': self.answer_consistency
        }


class AblationEvaluator:
    """
    Evaluator for ablation study results.

    Compares model performance across different ablation conditions
    to detect shortcut learning.
    """

    def __init__(self, baseline_results_path: str):
        """Initialize with baseline (normal) results."""
        self.baseline_results = self._load_results(baseline_results_path)
        self.ablation_results: Dict[str, Dict] = {}

    def _load_results(self, path: str) -> Dict[str, Dict]:
        """Load results from JSONL file."""
        results = {}
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                qid = data.get('question_id', '')
                results[qid] = data
        return results

    def _extract_binary_prediction(self, text: str) -> str:
        """Extract Yes/No prediction from model output."""
        text_lower = text.lower()

        # Check for explicit negative patterns first
        if any(p in text_lower for p in ['no missing', 'not missing', 'no components missing', 'properly installed']):
            return 'No'

        # Check for positive patterns
        if any(p in text_lower for p in ['missing', 'are missing', 'is missing']):
            return 'Yes'

        return 'Unknown'

    def add_ablation_results(self, name: str, results_path: str) -> None:
        """Add ablation test results for comparison."""
        self.ablation_results[name] = self._load_results(results_path)

    def evaluate_ablation(
        self,
        ablation_name: str,
        gt_path: str
    ) -> AblationResult:
        """
        Evaluate a single ablation test.

        Args:
            ablation_name: Name of the ablation (e.g., 'noise', 'cross')
            gt_path: Path to ground truth file

        Returns:
            AblationResult with evaluation metrics
        """
        if ablation_name not in self.ablation_results:
            raise ValueError(f"Ablation '{ablation_name}' not loaded")

        ablation_preds = self.ablation_results[ablation_name]

        # Load ground truth
        gt_data = {}
        with open(gt_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                qid = data.get('question_id', '')
                gt_data[qid] = data

        # Calculate metrics
        y_true = []
        y_pred = []
        yes_correct = 0
        yes_total = 0
        no_correct = 0
        no_total = 0

        for qid, pred in ablation_preds.items():
            # Find corresponding GT (handle ablation suffix)
            base_qid = qid.replace('_noise', '').replace('_shuffled', '').replace('_cross', '')
            gt = gt_data.get(base_qid)

            if gt is None:
                continue

            gt_label = gt.get('label', '').strip()
            pred_label = self._extract_binary_prediction(pred.get('text', ''))

            if gt_label in ['Yes', 'No']:
                y_true.append(1 if gt_label == 'Yes' else 0)
                y_pred.append(1 if pred_label == 'Yes' else 0)

                if gt_label == 'Yes':
                    yes_total += 1
                    if pred_label == 'Yes':
                        yes_correct += 1
                else:
                    no_total += 1
                    if pred_label == 'No':
                        no_correct += 1

        # Calculate accuracies
        if len(y_true) == 0:
            return AblationResult(
                test_name=ablation_name,
                num_samples=0,
                binary_accuracy=0.0,
                yes_accuracy=0.0,
                no_accuracy=0.0,
                accuracy_drop=0.0,
                bbox_iou=0.0,
                answer_consistency=0.0
            )

        binary_accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        yes_accuracy = yes_correct / max(yes_total, 1)
        no_accuracy = no_correct / max(no_total, 1)

        # Calculate accuracy drop vs baseline
        baseline_accuracy = self._calculate_baseline_accuracy(gt_path)
        accuracy_drop = baseline_accuracy - binary_accuracy

        # Calculate answer consistency with baseline
        consistency = self._calculate_consistency(ablation_preds, ablation_name)

        return AblationResult(
            test_name=ablation_name,
            num_samples=len(y_true),
            binary_accuracy=binary_accuracy,
            yes_accuracy=yes_accuracy,
            no_accuracy=no_accuracy,
            accuracy_drop=accuracy_drop,
            bbox_iou=0.0,  # TODO: Implement bbox evaluation
            answer_consistency=consistency
        )

    def _calculate_baseline_accuracy(self, gt_path: str) -> float:
        """Calculate baseline accuracy from normal test results."""
        gt_data = {}
        with open(gt_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                qid = data.get('question_id', '')
                gt_data[qid] = data

        correct = 0
        total = 0

        for qid, pred in self.baseline_results.items():
            gt = gt_data.get(qid)
            if gt is None:
                continue

            gt_label = gt.get('label', '').strip()
            pred_label = self._extract_binary_prediction(pred.get('text', ''))

            if gt_label in ['Yes', 'No']:
                total += 1
                if (gt_label == 'Yes' and pred_label == 'Yes') or \
                   (gt_label == 'No' and pred_label == 'No'):
                    correct += 1

        return correct / max(total, 1)

    def _calculate_consistency(self, ablation_preds: Dict, ablation_name: str) -> float:
        """Calculate how consistent ablation answers are with baseline."""
        matching = 0
        total = 0

        for qid, ablation_pred in ablation_preds.items():
            # Find baseline prediction
            base_qid = qid.replace('_noise', '').replace('_shuffled', '').replace('_cross', '')
            baseline_pred = self.baseline_results.get(base_qid)

            if baseline_pred is None:
                continue

            ablation_label = self._extract_binary_prediction(ablation_pred.get('text', ''))
            baseline_label = self._extract_binary_prediction(baseline_pred.get('text', ''))

            total += 1
            if ablation_label == baseline_label:
                matching += 1

        return matching / max(total, 1)

    def run_all_ablations(self, gt_path: str) -> Dict[str, AblationResult]:
        """Run evaluation on all loaded ablation tests."""
        results = {}
        for name in self.ablation_results:
            results[name] = self.evaluate_ablation(name, gt_path)
        return results

    def generate_report(self, gt_path: str) -> str:
        """Generate a comprehensive ablation study report."""
        results = self.run_all_ablations(gt_path)

        report = []
        report.append("=" * 80)
        report.append("ABLATION STUDY REPORT")
        report.append("=" * 80)
        report.append("")

        # Baseline
        baseline_acc = self._calculate_baseline_accuracy(gt_path)
        report.append(f"Baseline Accuracy: {baseline_acc:.2%}")
        report.append("")

        # Each ablation
        report.append("-" * 80)
        report.append("ABLATION RESULTS")
        report.append("-" * 80)

        for name, result in results.items():
            report.append(f"\n{name.upper()}")
            report.append(f"  Samples: {result.num_samples}")
            report.append(f"  Binary Accuracy: {result.binary_accuracy:.2%}")
            report.append(f"  Yes Accuracy: {result.yes_accuracy:.2%}")
            report.append(f"  No Accuracy: {result.no_accuracy:.2%}")
            report.append(f"  Accuracy Drop: {result.accuracy_drop:+.2%}")
            report.append(f"  Answer Consistency: {result.answer_consistency:.2%}")

        # Interpretation
        report.append("")
        report.append("-" * 80)
        report.append("INTERPRETATION")
        report.append("-" * 80)

        # Check for shortcut learning
        shortcut_indicators = []

        for name, result in results.items():
            if name == 'noise' and result.accuracy_drop < 0.1:
                shortcut_indicators.append(
                    f"  - {name}: Model performs similarly on random noise "
                    f"(drop: {result.accuracy_drop:.2%}). "
                    "This suggests the model may not be using point cloud features."
                )

            if name == 'cross' and result.answer_consistency > 0.8:
                shortcut_indicators.append(
                    f"  - {name}: High consistency ({result.answer_consistency:.2%}) "
                    "with mismatched point clouds. "
                    "This suggests text-based shortcuts."
                )

        if shortcut_indicators:
            report.append("\nWARNING: Potential shortcut learning detected!")
            report.extend(shortcut_indicators)
        else:
            report.append("\nNo obvious shortcut learning detected.")
            report.append("Model appears to use point cloud information appropriately.")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def main():
    """Example usage of ablation evaluator."""
    import argparse

    parser = argparse.ArgumentParser(description='Run ablation study evaluation')
    parser.add_argument('--baseline', required=True, help='Path to baseline predictions')
    parser.add_argument('--gt', required=True, help='Path to ground truth')
    parser.add_argument('--ablations', nargs='+', help='Paths to ablation predictions (name:path)')
    parser.add_argument('--output', default='ablation_report.txt', help='Output report path')

    args = parser.parse_args()

    evaluator = AblationEvaluator(args.baseline)

    if args.ablations:
        for ablation in args.ablations:
            name, path = ablation.split(':')
            evaluator.add_ablation_results(name, path)

    report = evaluator.generate_report(args.gt)
    print(report)

    with open(args.output, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {args.output}")


if __name__ == '__main__':
    main()
