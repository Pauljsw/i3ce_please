#!/usr/bin/env python3
"""
Evaluation script for scaffold missing detection.

This script evaluates model predictions against ground truth and computes:
- Binary Accuracy (Yes/No classification)
- BBox IoU@0.5 (3D bounding box intersection over union)
- Per-category metrics
- Ablation study results

Usage:
    python tools/evaluate_scaffold.py \
        --predictions ./outputs/test_answers.jsonl \
        --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
        --output-dir ./evaluation_results
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np


@dataclass
class BBox3D:
    """3D bounding box represented by 8 corners."""
    corners: np.ndarray  # Shape: (8, 3)

    @classmethod
    def from_list(cls, corners_list: List[List[float]]) -> 'BBox3D':
        """Create from list of 8 corners."""
        return cls(corners=np.array(corners_list))

    def get_axis_aligned_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned min/max bounds."""
        min_bound = self.corners.min(axis=0)
        max_bound = self.corners.max(axis=0)
        return min_bound, max_bound

    def volume(self) -> float:
        """Calculate volume using axis-aligned approximation."""
        min_b, max_b = self.get_axis_aligned_bounds()
        dims = max_b - min_b
        return float(np.prod(dims))


def compute_iou_3d(box1: BBox3D, box2: BBox3D) -> float:
    """
    Compute 3D IoU between two bounding boxes.
    Uses axis-aligned approximation for simplicity.
    """
    min1, max1 = box1.get_axis_aligned_bounds()
    min2, max2 = box2.get_axis_aligned_bounds()

    # Intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    # Check if there's any intersection
    if np.any(inter_max <= inter_min):
        return 0.0

    inter_dims = inter_max - inter_min
    inter_vol = float(np.prod(inter_dims))

    # Union
    vol1 = box1.volume()
    vol2 = box2.volume()
    union_vol = vol1 + vol2 - inter_vol

    if union_vol <= 0:
        return 0.0

    return inter_vol / union_vol


def parse_answer_text(text: str) -> Tuple[str, List[List[List[float]]]]:
    """
    Parse model answer to extract Yes/No label and bounding boxes.

    Returns:
        (label, bboxes) where label is 'Yes' or 'No' and bboxes is list of 8-corner boxes
    """
    text_lower = text.lower().strip()

    # Determine Yes/No label
    if text_lower.startswith('yes') or 'missing' in text_lower or 'detected' in text_lower:
        if 'no missing' in text_lower or 'no defect' in text_lower or 'properly installed' in text_lower:
            label = 'No'
        else:
            label = 'Yes'
    elif text_lower.startswith('no'):
        label = 'No'
    else:
        # Try to infer from content
        if any(word in text_lower for word in ['missing', 'defect', 'absent', 'lack']):
            label = 'Yes'
        else:
            label = 'No'

    # Extract bounding boxes
    # Pattern: [[x,y,z], [x,y,z], ...] or similar
    bboxes = []

    # Try to find bbox patterns
    bbox_pattern = r'\[\s*\[\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*\](?:\s*,\s*\[\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*\]){7}\s*\]'

    matches = re.findall(bbox_pattern, text)

    for match in matches:
        try:
            bbox = json.loads(match.replace(' ', ''))
            if len(bbox) == 8 and all(len(corner) == 3 for corner in bbox):
                bboxes.append(bbox)
        except json.JSONDecodeError:
            continue

    return label, bboxes


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    binary_accuracy: float
    bbox_iou_mean: float
    bbox_iou_at_05: float  # Percentage of boxes with IoU >= 0.5
    bbox_iou_at_025: float  # Percentage of boxes with IoU >= 0.25

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    precision: float
    recall: float
    f1_score: float

    num_samples: int
    num_predicted_boxes: int
    num_gt_boxes: int

    per_task_accuracy: Dict[str, float]


def evaluate_predictions(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5
) -> EvaluationResult:
    """
    Evaluate predictions against ground truth.

    Args:
        predictions: List of prediction dicts with 'question_id' and 'text'
        ground_truth: List of GT dicts with 'question_id', 'label', 'bboxes', 'task_type'
        iou_threshold: IoU threshold for counting as correct detection

    Returns:
        EvaluationResult with all metrics
    """
    # Build GT lookup
    gt_lookup = {gt['question_id']: gt for gt in ground_truth}

    # Metrics accumulators
    binary_correct = 0
    total_samples = 0

    tp, tn, fp, fn = 0, 0, 0, 0

    all_ious = []
    task_correct = defaultdict(int)
    task_total = defaultdict(int)

    num_pred_boxes = 0
    num_gt_boxes = 0

    for pred in predictions:
        qid = pred.get('question_id')
        if qid not in gt_lookup:
            continue

        gt = gt_lookup[qid]
        gt_label = gt.get('label', 'No')
        gt_bboxes = gt.get('bboxes', [])
        task_type = gt.get('task_type', 'unknown')

        # Parse prediction
        pred_text = pred.get('text', '')
        pred_label, pred_bboxes = parse_answer_text(pred_text)

        total_samples += 1
        task_total[task_type] += 1

        # Binary accuracy
        if pred_label == gt_label:
            binary_correct += 1
            task_correct[task_type] += 1

        # Confusion matrix
        if gt_label == 'Yes':
            if pred_label == 'Yes':
                tp += 1
            else:
                fn += 1
        else:  # gt_label == 'No'
            if pred_label == 'No':
                tn += 1
            else:
                fp += 1

        # BBox IoU (only for Yes cases with boxes)
        num_pred_boxes += len(pred_bboxes)
        num_gt_boxes += len(gt_bboxes)

        if gt_bboxes and pred_bboxes:
            # Greedy matching
            gt_boxes = [BBox3D.from_list(b) for b in gt_bboxes]
            pred_boxes = [BBox3D.from_list(b) for b in pred_bboxes]

            matched_gt = set()

            for pred_box in pred_boxes:
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue

                    iou = compute_iou_3d(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                all_ious.append(best_iou)
                if best_gt_idx >= 0 and best_iou >= iou_threshold:
                    matched_gt.add(best_gt_idx)

    # Calculate final metrics
    binary_accuracy = (binary_correct / total_samples * 100) if total_samples > 0 else 0.0

    # IoU metrics
    bbox_iou_mean = (np.mean(all_ious) * 100) if all_ious else 0.0
    bbox_iou_at_05 = (sum(1 for iou in all_ious if iou >= 0.5) / len(all_ious) * 100) if all_ious else 0.0
    bbox_iou_at_025 = (sum(1 for iou in all_ious if iou >= 0.25) / len(all_ious) * 100) if all_ious else 0.0

    # Precision, Recall, F1
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Per-task accuracy
    per_task_accuracy = {
        task: (task_correct[task] / task_total[task] * 100) if task_total[task] > 0 else 0.0
        for task in task_total.keys()
    }

    return EvaluationResult(
        binary_accuracy=binary_accuracy,
        bbox_iou_mean=bbox_iou_mean,
        bbox_iou_at_05=bbox_iou_at_05,
        bbox_iou_at_025=bbox_iou_at_025,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        num_samples=total_samples,
        num_predicted_boxes=num_pred_boxes,
        num_gt_boxes=num_gt_boxes,
        per_task_accuracy=per_task_accuracy
    )


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def print_evaluation_report(result: EvaluationResult, title: str = "Evaluation Results"):
    """Print formatted evaluation report."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

    print("\n[Binary Classification]")
    print(f"  Accuracy:  {result.binary_accuracy:.2f}%")
    print(f"  Precision: {result.precision:.2f}%")
    print(f"  Recall:    {result.recall:.2f}%")
    print(f"  F1 Score:  {result.f1_score:.2f}%")

    print("\n[Confusion Matrix]")
    print(f"  True Positive:  {result.true_positive}")
    print(f"  True Negative:  {result.true_negative}")
    print(f"  False Positive: {result.false_positive}")
    print(f"  False Negative: {result.false_negative}")

    print("\n[BBox Detection]")
    print(f"  Mean IoU:     {result.bbox_iou_mean:.2f}%")
    print(f"  IoU@0.50:     {result.bbox_iou_at_05:.2f}%")
    print(f"  IoU@0.25:     {result.bbox_iou_at_025:.2f}%")
    print(f"  Pred boxes:   {result.num_predicted_boxes}")
    print(f"  GT boxes:     {result.num_gt_boxes}")

    print("\n[Per-Task Accuracy]")
    for task, acc in sorted(result.per_task_accuracy.items()):
        print(f"  {task}: {acc:.2f}%")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate scaffold detection model')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSONL file')
    parser.add_argument('--ground-truth', type=str, required=True,
                       help='Path to ground truth JSONL file')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--ablation-dirs', type=str, nargs='*',
                       help='Paths to ablation prediction files for comparison')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading predictions from: {args.predictions}")
    predictions = load_jsonl(args.predictions)

    print(f"Loading ground truth from: {args.ground_truth}")
    ground_truth = load_jsonl(args.ground_truth)

    print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth entries")

    # Main evaluation
    result = evaluate_predictions(predictions, ground_truth)
    print_evaluation_report(result, "Main Model Evaluation")

    # Save results
    result_dict = asdict(result)
    result_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nSaved results to: {result_path}")

    # Ablation comparison if provided
    if args.ablation_dirs:
        ablation_results = {'main': result_dict}

        for ablation_path in args.ablation_dirs:
            if os.path.exists(ablation_path):
                ablation_name = os.path.basename(os.path.dirname(ablation_path))
                ablation_preds = load_jsonl(ablation_path)
                ablation_result = evaluate_predictions(ablation_preds, ground_truth)

                print_evaluation_report(ablation_result, f"Ablation: {ablation_name}")
                ablation_results[ablation_name] = asdict(ablation_result)

        # Save comparison
        comparison_path = os.path.join(args.output_dir, 'ablation_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(ablation_results, f, indent=2)
        print(f"\nSaved ablation comparison to: {comparison_path}")

    # Generate visualization data for paper_visualization.py
    viz_data = {
        'main': {
            'Binary Accuracy': result.binary_accuracy,
            'BBox IoU@0.5': result.bbox_iou_at_05,
            'Precision': result.precision,
            'Recall': result.recall,
            'F1 Score': result.f1_score
        }
    }

    viz_path = os.path.join(args.output_dir, 'visualization_data.json')
    with open(viz_path, 'w') as f:
        json.dump(viz_data, f, indent=2)
    print(f"Saved visualization data to: {viz_path}")

    print("\n[Done] Evaluation complete!")


if __name__ == '__main__':
    main()
