#!/usr/bin/env python3
"""
Rigorous Academic Evaluation for Scaffold Missing Detection - V2

Task-Type Aware Evaluation:
- Different metrics apply to different question types
- Per-task breakdown for all metrics

Question Types and Applicable Metrics:
┌─────────────────────────────────┬─────────┬───────┬──────────┬──────────┐
│ Task Type                       │ Binary  │ Count │ BBox IoU │ CompType │
├─────────────────────────────────┼─────────┼───────┼──────────┼──────────┤
│ missing_detection_summary       │    ✓    │   ✓   │    ✓     │    ✓     │
│ missing_detection_floor         │    ✓    │   ✓   │    ✓     │    -     │
│ missing_detection_bay           │    ✓    │   ✓   │    ✓     │    -     │
│ missing_detection_specific      │    ✓    │   -   │    ✓     │    -     │
│ missing_detection_vertical_sum  │    ✓    │   ✓   │    ✓     │    -     │
│ missing_detection_horizontal_sum│    ✓    │   ✓   │    ✓     │    -     │
└─────────────────────────────────┴─────────┴───────┴──────────┴──────────┘

Usage:
    python tools/evaluate_scaffold_rigorous.py \
        --predictions ./outputs/test_answers.jsonl \
        --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
        --output-dir ./evaluation_results
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

import numpy as np


# ============================================================================
# Task Type Definitions
# ============================================================================

# Define which metrics apply to which task types
TASK_METRIC_APPLICABILITY = {
    'missing_detection_summary': {
        'binary': True,
        'count': True,
        'bbox': True,
        'component_type': True,
        'description': 'Overall scaffold missing detection'
    },
    'missing_detection_floor': {
        'binary': True,
        'count': True,
        'bbox': True,
        'component_type': False,
        'description': 'Floor-level missing detection'
    },
    'missing_detection_bay': {
        'binary': True,
        'count': True,
        'bbox': True,
        'component_type': False,
        'description': 'Bay-level missing detection'
    },
    'missing_detection_specific': {
        'binary': True,
        'count': False,  # Always 0 or 1, not meaningful
        'bbox': True,    # Single bbox check
        'component_type': False,
        'description': 'Specific component presence check'
    },
    'missing_detection_vertical_summary': {
        'binary': True,
        'count': True,
        'bbox': True,
        'component_type': False,
        'description': 'Vertical post missing summary'
    },
    'missing_detection_horizontal_summary': {
        'binary': True,
        'count': True,
        'bbox': True,
        'component_type': False,
        'description': 'Horizontal beam missing summary'
    }
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BBox3D:
    """3D bounding box represented by 8 corners."""
    corners: np.ndarray

    @classmethod
    def from_list(cls, corners_list: List[List[float]]) -> Optional['BBox3D']:
        try:
            arr = np.array(corners_list)
            if arr.shape == (8, 3):
                return cls(corners=arr)
        except:
            pass
        return None

    def get_axis_aligned_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.corners.min(axis=0), self.corners.max(axis=0)

    def volume(self) -> float:
        min_b, max_b = self.get_axis_aligned_bounds()
        dims = max_b - min_b
        return float(np.prod(np.maximum(dims, 0)))

    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)


@dataclass
class TaskMetrics:
    """Metrics for a specific task type."""
    task_type: str
    description: str = ""

    # Sample counts
    total: int = 0
    num_yes_gt: int = 0
    num_no_gt: int = 0

    # Binary Classification
    binary_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    # Count (if applicable)
    count_applicable: bool = False
    count_exact_accuracy: float = 0.0
    count_tolerance1_accuracy: float = 0.0
    count_mae: float = 0.0
    count_samples: int = 0

    # BBox (if applicable)
    bbox_applicable: bool = True
    bbox_iou_mean: float = 0.0
    bbox_iou_at_50: float = 0.0
    bbox_iou_at_25: float = 0.0
    bbox_detection_rate: float = 0.0
    num_gt_boxes: int = 0
    num_pred_boxes: int = 0


@dataclass
class OverallMetrics:
    """Aggregated metrics across all task types."""
    # Overall Binary
    binary_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Overall Count (excluding specific)
    count_exact_accuracy: float = 0.0
    count_mae: float = 0.0

    # Overall BBox
    bbox_iou_mean: float = 0.0
    bbox_iou_at_50: float = 0.0
    bbox_iou_at_25: float = 0.0
    bbox_detection_rate: float = 0.0

    # Component Type (summary only)
    component_detection_accuracy: float = 0.0
    vertical_detection_rate: float = 0.0
    horizontal_detection_rate: float = 0.0
    platform_detection_rate: float = 0.0

    # Sample counts
    num_samples: int = 0
    num_yes_gt: int = 0
    num_no_gt: int = 0
    num_gt_boxes: int = 0
    num_pred_boxes: int = 0

    # Confusion matrix
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    overall_metrics: OverallMetrics
    per_task_metrics: Dict[str, TaskMetrics]
    data_leakage_detected: bool = False
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Parsing Functions
# ============================================================================

def parse_binary_from_prediction(pred_text: str, task_type: str) -> str:
    """
    Parse Yes/No from model prediction text.

    For specific questions: Yes = present, No = missing
    For all others: Yes = missing exists, No = no missing
    """
    text_lower = pred_text.lower().strip()

    # For specific component questions, the meaning is reversed
    if task_type == 'missing_detection_specific':
        if text_lower.startswith('yes'):
            return 'Yes'
        elif text_lower.startswith('no'):
            return 'No'
        if 'missing' in text_lower or 'absent' in text_lower:
            return 'No'
        if 'present' in text_lower or 'exists' in text_lower or 'there is' in text_lower:
            return 'Yes'
        return 'No'

    # For all other question types
    if text_lower.startswith('yes'):
        if 'no missing' in text_lower[:50] or 'no defect' in text_lower[:50]:
            return 'No'
        return 'Yes'
    elif text_lower.startswith('no'):
        if 'missing' in text_lower[3:30] and 'no missing' not in text_lower:
            return 'Yes'
        return 'No'

    # Inference from content
    negative_phrases = [
        'no missing', 'no defect', 'properly installed',
        'all present', 'all components are present', 'complete structure'
    ]
    for phrase in negative_phrases:
        if phrase in text_lower:
            return 'No'

    positive_phrases = ['missing', 'detected', 'found', 'absent', 'lack', 'defect']
    for phrase in positive_phrases:
        if phrase in text_lower:
            return 'Yes'

    return 'No'


def parse_missing_count_from_text(text: str) -> int:
    """Parse the number of missing components from answer text."""
    text_lower = text.lower()

    # Pattern 1: "Missing components detected (N total)"
    match = re.search(r'missing[^(]*\((\d+)\s*total\)', text_lower)
    if match:
        return int(match.group(1))

    # Pattern 2: "N component(s) are missing"
    match = re.search(r'(\d+)\s*component[s]?\s*(?:are|is)?\s*missing', text_lower)
    if match:
        return int(match.group(1))

    # Pattern 3: Count bullet points
    bullet_count = len(re.findall(r'^-\s+(?:vertical|horizontal|platform)', text_lower, re.MULTILINE))
    if bullet_count > 0:
        return bullet_count

    # Pattern 4: "Missing: N"
    match = re.search(r'missing:\s*(\d+)', text_lower)
    if match:
        return int(match.group(1))

    return 0


def parse_component_types_from_text(text: str) -> Dict[str, int]:
    """Parse component types mentioned as missing from text."""
    counts = {'vertical': 0, 'horizontal': 0, 'platform': 0}
    text_lower = text.lower()

    counts['vertical'] = len(re.findall(r'-\s*vertical\s+post', text_lower))
    counts['horizontal'] = len(re.findall(r'-\s*horizontal\s+beam', text_lower))
    counts['platform'] = len(re.findall(r'-\s*platform\s+at', text_lower))

    return counts


def parse_bboxes_from_text(text: str) -> List[List[List[float]]]:
    """Extract 3D bounding boxes from text."""
    bboxes = []
    bbox_pattern = r'\[\s*\[[\s\d.,-]+\](?:\s*,\s*\[[\s\d.,-]+\]){7}\s*\]'
    matches = re.findall(bbox_pattern, text)

    for match in matches:
        try:
            clean = re.sub(r'\s+', '', match)
            bbox = json.loads(clean)
            if len(bbox) == 8 and all(len(corner) == 3 for corner in bbox):
                bboxes.append(bbox)
        except (json.JSONDecodeError, ValueError):
            continue

    return bboxes


# ============================================================================
# IoU Computation
# ============================================================================

def compute_iou_3d(box1: BBox3D, box2: BBox3D) -> float:
    """Compute 3D IoU using axis-aligned bounding box approximation."""
    min1, max1 = box1.get_axis_aligned_bounds()
    min2, max2 = box2.get_axis_aligned_bounds()

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    if np.any(inter_max <= inter_min):
        return 0.0

    inter_vol = float(np.prod(inter_max - inter_min))
    union_vol = box1.volume() + box2.volume() - inter_vol

    return inter_vol / union_vol if union_vol > 0 else 0.0


def match_bboxes(
    pred_bboxes: List[List[List[float]]],
    gt_bboxes: List[List[List[float]]],
    iou_threshold: float = 0.25
) -> Tuple[List[float], int, int]:
    """Match predicted boxes to ground truth."""
    if not gt_bboxes:
        return [], 0, len(pred_bboxes)

    if not pred_bboxes:
        return [], 0, 0

    pred_boxes = [BBox3D.from_list(b) for b in pred_bboxes]
    gt_boxes = [BBox3D.from_list(b) for b in gt_bboxes]

    pred_boxes = [b for b in pred_boxes if b is not None]
    gt_boxes = [b for b in gt_boxes if b is not None]

    if not pred_boxes or not gt_boxes:
        return [], 0, len(pred_bboxes)

    matched_gt = set()
    all_ious = []
    num_tp = 0

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
            num_tp += 1

    num_fp = len(pred_boxes) - num_tp
    return all_ious, num_tp, num_fp


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_detailed(
    predictions: List[Dict],
    ground_truth: List[Dict]
) -> Tuple[OverallMetrics, Dict[str, TaskMetrics]]:
    """
    Compute comprehensive evaluation metrics with task-type awareness.
    """
    gt_lookup = {gt['question_id']: gt for gt in ground_truth}

    # Initialize per-task accumulators
    task_data = defaultdict(lambda: {
        'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
        'count_errors': [], 'count_samples': 0,
        'count_exact': 0, 'count_tol1': 0,
        'ious': [], 'tp_boxes': 0, 'gt_boxes': 0, 'pred_boxes': 0,
        'total': 0, 'num_yes_gt': 0, 'num_no_gt': 0
    })

    # Component detection (summary only)
    comp_gt_counts = {'vertical': 0, 'horizontal': 0, 'platform': 0}
    comp_pred_correct = {'vertical': 0, 'horizontal': 0, 'platform': 0}

    for pred in predictions:
        qid = pred.get('question_id')
        if qid not in gt_lookup:
            continue

        gt = gt_lookup[qid]
        gt_label = gt.get('label', 'No')
        gt_bboxes = gt.get('bboxes', [])
        gt_answer_text = gt.get('text', '')
        task_type = gt.get('task_type', 'unknown')

        pred_text = pred.get('text', '')
        pred_label = parse_binary_from_prediction(pred_text, task_type)
        pred_bboxes = parse_bboxes_from_text(pred_text)

        td = task_data[task_type]
        td['total'] += 1

        # ============ Binary Classification ============
        if gt_label == 'Yes':
            td['num_yes_gt'] += 1
            if pred_label == 'Yes':
                td['tp'] += 1
            else:
                td['fn'] += 1
        else:
            td['num_no_gt'] += 1
            if pred_label == 'No':
                td['tn'] += 1
            else:
                td['fp'] += 1

        # ============ Count Accuracy ============
        # Only for applicable task types and Yes cases
        task_config = TASK_METRIC_APPLICABILITY.get(task_type, {})
        if task_config.get('count', False) and gt_label == 'Yes':
            gt_count = parse_missing_count_from_text(gt_answer_text)
            pred_count = parse_missing_count_from_text(pred_text)

            if gt_count > 0:
                td['count_samples'] += 1
                error = abs(pred_count - gt_count)
                td['count_errors'].append(error)
                if error == 0:
                    td['count_exact'] += 1
                if error <= 1:
                    td['count_tol1'] += 1

        # ============ Component Type Detection ============
        # Only for summary with Yes label
        if task_config.get('component_type', False) and gt_label == 'Yes':
            gt_comp = parse_component_types_from_text(gt_answer_text)
            pred_comp = parse_component_types_from_text(pred_text)

            for comp_type in ['vertical', 'horizontal', 'platform']:
                gt_c = gt_comp[comp_type]
                pred_c = pred_comp[comp_type]
                if gt_c > 0:
                    comp_gt_counts[comp_type] += gt_c
                    comp_pred_correct[comp_type] += min(pred_c, gt_c)

        # ============ BBox IoU ============
        td['pred_boxes'] += len(pred_bboxes)
        td['gt_boxes'] += len(gt_bboxes)

        if gt_bboxes:
            ious, tp_boxes, fp_boxes = match_bboxes(pred_bboxes, gt_bboxes)
            td['ious'].extend(ious)
            td['tp_boxes'] += tp_boxes

    # ============ Calculate Per-Task Metrics ============
    per_task_metrics = {}

    for task_type, td in task_data.items():
        task_config = TASK_METRIC_APPLICABILITY.get(task_type, {})

        tm = TaskMetrics(
            task_type=task_type,
            description=task_config.get('description', ''),
            total=td['total'],
            num_yes_gt=td['num_yes_gt'],
            num_no_gt=td['num_no_gt'],
            tp=td['tp'],
            tn=td['tn'],
            fp=td['fp'],
            fn=td['fn'],
            count_applicable=task_config.get('count', False),
            bbox_applicable=task_config.get('bbox', True)
        )

        # Binary metrics
        n = td['total']
        if n > 0:
            tm.binary_accuracy = (td['tp'] + td['tn']) / n * 100
        if (td['tp'] + td['fp']) > 0:
            tm.precision = td['tp'] / (td['tp'] + td['fp']) * 100
        if (td['tp'] + td['fn']) > 0:
            tm.recall = td['tp'] / (td['tp'] + td['fn']) * 100
        if tm.precision + tm.recall > 0:
            tm.f1_score = 2 * tm.precision * tm.recall / (tm.precision + tm.recall)

        # Count metrics
        if td['count_samples'] > 0:
            tm.count_exact_accuracy = td['count_exact'] / td['count_samples'] * 100
            tm.count_tolerance1_accuracy = td['count_tol1'] / td['count_samples'] * 100
            tm.count_samples = td['count_samples']
        if td['count_errors']:
            tm.count_mae = np.mean(td['count_errors'])

        # BBox metrics
        tm.num_gt_boxes = td['gt_boxes']
        tm.num_pred_boxes = td['pred_boxes']
        if td['ious']:
            tm.bbox_iou_mean = np.mean(td['ious']) * 100
            tm.bbox_iou_at_50 = sum(1 for iou in td['ious'] if iou >= 0.5) / len(td['ious']) * 100
            tm.bbox_iou_at_25 = sum(1 for iou in td['ious'] if iou >= 0.25) / len(td['ious']) * 100
        if td['gt_boxes'] > 0:
            tm.bbox_detection_rate = td['tp_boxes'] / td['gt_boxes'] * 100

        per_task_metrics[task_type] = tm

    # ============ Calculate Overall Metrics ============
    overall = OverallMetrics()

    # Aggregate binary metrics
    total_tp = sum(td['tp'] for td in task_data.values())
    total_tn = sum(td['tn'] for td in task_data.values())
    total_fp = sum(td['fp'] for td in task_data.values())
    total_fn = sum(td['fn'] for td in task_data.values())
    total_n = sum(td['total'] for td in task_data.values())

    overall.tp, overall.tn, overall.fp, overall.fn = total_tp, total_tn, total_fp, total_fn
    overall.num_samples = total_n
    overall.num_yes_gt = sum(td['num_yes_gt'] for td in task_data.values())
    overall.num_no_gt = sum(td['num_no_gt'] for td in task_data.values())

    if total_n > 0:
        overall.binary_accuracy = (total_tp + total_tn) / total_n * 100
    if (total_tp + total_fp) > 0:
        overall.precision = total_tp / (total_tp + total_fp) * 100
    if (total_tp + total_fn) > 0:
        overall.recall = total_tp / (total_tp + total_fn) * 100
    if overall.precision + overall.recall > 0:
        overall.f1_score = 2 * overall.precision * overall.recall / (overall.precision + overall.recall)

    # Aggregate count metrics (excluding 'specific' type)
    count_exact_total = 0
    count_samples_total = 0
    all_count_errors = []
    for task_type, td in task_data.items():
        if TASK_METRIC_APPLICABILITY.get(task_type, {}).get('count', False):
            count_exact_total += td['count_exact']
            count_samples_total += td['count_samples']
            all_count_errors.extend(td['count_errors'])

    if count_samples_total > 0:
        overall.count_exact_accuracy = count_exact_total / count_samples_total * 100
    if all_count_errors:
        overall.count_mae = np.mean(all_count_errors)

    # Aggregate bbox metrics
    all_ious = []
    total_tp_boxes = 0
    total_gt_boxes = 0
    total_pred_boxes = 0
    for td in task_data.values():
        all_ious.extend(td['ious'])
        total_tp_boxes += td['tp_boxes']
        total_gt_boxes += td['gt_boxes']
        total_pred_boxes += td['pred_boxes']

    overall.num_gt_boxes = total_gt_boxes
    overall.num_pred_boxes = total_pred_boxes

    if all_ious:
        overall.bbox_iou_mean = np.mean(all_ious) * 100
        overall.bbox_iou_at_50 = sum(1 for iou in all_ious if iou >= 0.5) / len(all_ious) * 100
        overall.bbox_iou_at_25 = sum(1 for iou in all_ious if iou >= 0.25) / len(all_ious) * 100
    if total_gt_boxes > 0:
        overall.bbox_detection_rate = total_tp_boxes / total_gt_boxes * 100

    # Component detection rates
    for comp_type in ['vertical', 'horizontal', 'platform']:
        if comp_gt_counts[comp_type] > 0:
            rate = comp_pred_correct[comp_type] / comp_gt_counts[comp_type] * 100
            setattr(overall, f'{comp_type}_detection_rate', rate)

    total_comp_gt = sum(comp_gt_counts.values())
    total_comp_correct = sum(comp_pred_correct.values())
    if total_comp_gt > 0:
        overall.component_detection_accuracy = total_comp_correct / total_comp_gt * 100

    return overall, per_task_metrics


# ============================================================================
# Data Leakage Check
# ============================================================================

def check_data_leakage(overall: OverallMetrics) -> Tuple[bool, List[str]]:
    """Check for data leakage indicators."""
    warnings = []
    leakage_detected = False

    if overall.binary_accuracy > 95 and overall.bbox_iou_at_25 < 5:
        leakage_detected = True
        warnings.append(
            f"DATA LEAKAGE SUSPECTED: Binary accuracy {overall.binary_accuracy:.1f}% "
            f"but BBox IoU@0.25 only {overall.bbox_iou_at_25:.1f}%"
        )

    if overall.binary_accuracy > 99:
        warnings.append(f"WARNING: Binary accuracy {overall.binary_accuracy:.1f}% is suspiciously high")

    return leakage_detected, warnings


# ============================================================================
# Reporting
# ============================================================================

def print_evaluation_report(report: EvaluationReport):
    """Print comprehensive evaluation report."""
    o = report.overall_metrics

    print("\n" + "=" * 80)
    print(" SCAFFOLD MISSING DETECTION - TASK-AWARE EVALUATION REPORT")
    print(f" Generated: {report.timestamp}")
    print("=" * 80)

    if report.data_leakage_detected:
        print("\n⚠️  DATA LEAKAGE DETECTED!")
    else:
        print("\n✓ No obvious data leakage detected")

    # ============ OVERALL METRICS ============
    print("\n" + "=" * 80)
    print(" OVERALL METRICS")
    print("=" * 80)

    print("\n[1] Binary Classification (All Tasks)")
    print("-" * 50)
    print(f"  Accuracy:  {o.binary_accuracy:.2f}%")
    print(f"  Precision: {o.precision:.2f}%")
    print(f"  Recall:    {o.recall:.2f}%")
    print(f"  F1 Score:  {o.f1_score:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Yes    No")
    print(f"    Actual Yes {o.tp:4d}  {o.fn:4d}")
    print(f"           No  {o.fp:4d}  {o.tn:4d}")
    print(f"\n  GT Distribution: Yes={o.num_yes_gt}, No={o.num_no_gt}")

    print("\n[2] Count Accuracy (Excludes 'specific' type)")
    print("-" * 50)
    print(f"  Exact Match: {o.count_exact_accuracy:.2f}%")
    print(f"  Mean Abs Error: {o.count_mae:.2f}")

    print("\n[3] BBox Grounding (All Tasks)")
    print("-" * 50)
    print(f"  Mean IoU:       {o.bbox_iou_mean:.2f}%")
    print(f"  IoU@0.50:       {o.bbox_iou_at_50:.2f}%")
    print(f"  IoU@0.25:       {o.bbox_iou_at_25:.2f}%")
    print(f"  Detection Rate: {o.bbox_detection_rate:.2f}%")
    print(f"  (GT Boxes: {o.num_gt_boxes}, Pred Boxes: {o.num_pred_boxes})")

    print("\n[4] Component Type Detection (Summary only)")
    print("-" * 50)
    print(f"  Overall:    {o.component_detection_accuracy:.2f}%")
    print(f"  Vertical:   {o.vertical_detection_rate:.2f}%")
    print(f"  Horizontal: {o.horizontal_detection_rate:.2f}%")
    print(f"  Platform:   {o.platform_detection_rate:.2f}%")

    # ============ PER-TASK METRICS ============
    print("\n" + "=" * 80)
    print(" PER-TASK BREAKDOWN")
    print("=" * 80)

    for task_type in sorted(report.per_task_metrics.keys()):
        tm = report.per_task_metrics[task_type]
        config = TASK_METRIC_APPLICABILITY.get(task_type, {})

        print(f"\n[{task_type}]")
        print(f"  Description: {tm.description}")
        print(f"  Samples: {tm.total} (Yes={tm.num_yes_gt}, No={tm.num_no_gt})")
        print("-" * 40)

        # Binary (always applicable)
        print(f"  Binary Accuracy: {tm.binary_accuracy:.2f}%")
        print(f"  Precision: {tm.precision:.2f}%, Recall: {tm.recall:.2f}%, F1: {tm.f1_score:.2f}%")

        # Count (if applicable)
        if config.get('count', False):
            print(f"  Count Exact: {tm.count_exact_accuracy:.2f}%, MAE: {tm.count_mae:.2f}")
        else:
            print(f"  Count: N/A (not applicable for this task type)")

        # BBox (if applicable)
        if config.get('bbox', True):
            print(f"  BBox IoU Mean: {tm.bbox_iou_mean:.2f}%, IoU@0.25: {tm.bbox_iou_at_25:.2f}%")
            print(f"  BBox Detection Rate: {tm.bbox_detection_rate:.2f}% (GT: {tm.num_gt_boxes}, Pred: {tm.num_pred_boxes})")

    # Warnings
    if report.warnings:
        print("\n" + "=" * 80)
        print(" ⚠️  WARNINGS")
        print("=" * 80)
        for w in report.warnings:
            print(f"  • {w}")

    print("\n" + "=" * 80)


def save_results(report: EvaluationReport, output_dir: str):
    """Save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON report
    report_dict = {
        'timestamp': report.timestamp,
        'data_leakage_detected': report.data_leakage_detected,
        'overall_metrics': asdict(report.overall_metrics),
        'per_task_metrics': {k: asdict(v) for k, v in report.per_task_metrics.items()},
        'warnings': report.warnings
    }

    with open(os.path.join(output_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report_dict, f, indent=2)

    print(f"\nResults saved to: {output_dir}/evaluation_report.json")


# ============================================================================
# Utility
# ============================================================================

def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Task-Aware Rigorous Evaluation')
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--ground-truth', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./evaluation_results')

    args = parser.parse_args()

    print("=" * 80)
    print(" TASK-AWARE RIGOROUS EVALUATION V2")
    print("=" * 80)

    print(f"\nLoading predictions: {args.predictions}")
    predictions = load_jsonl(args.predictions)

    print(f"Loading ground truth: {args.ground_truth}")
    ground_truth = load_jsonl(args.ground_truth)

    print(f"Loaded {len(predictions)} predictions, {len(ground_truth)} ground truth")

    print("\nRunning evaluation...")
    overall, per_task = evaluate_detailed(predictions, ground_truth)

    leakage_detected, warnings = check_data_leakage(overall)

    report = EvaluationReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        overall_metrics=overall,
        per_task_metrics=per_task,
        data_leakage_detected=leakage_detected,
        warnings=warnings
    )

    print_evaluation_report(report)
    save_results(report, args.output_dir)

    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()