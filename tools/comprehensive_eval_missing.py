#!/usr/bin/env python3
"""
Comprehensive Evaluation for Scaffold Missing Detection
Evaluates tasks with proper question-type filtering based on actual data generation patterns
"""

import os
import re
import json
import ast
import argparse
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class ComprehensiveMissingEvaluator:
    """Comprehensive evaluator for all missing detection tasks"""

    def __init__(self, gt_file: str, pred_file: str):
        """Initialize evaluator with GT and prediction files"""
        self.gt_data = self.load_jsonl(gt_file)
        self.pred_data = self.load_jsonl(pred_file)

        # Match by question_id
        self.matched_pairs = self.match_predictions()

        print(f"✅ Loaded {len(self.gt_data)} GT samples")
        print(f"✅ Loaded {len(self.pred_data)} predictions")
        print(f"✅ Matched {len(self.matched_pairs)} pairs")

    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def match_predictions(self) -> List[Tuple[Dict, Dict]]:
        """Match GT and predictions by question_id"""
        pred_dict = {item['question_id']: item for item in self.pred_data}

        matched = []
        for gt in self.gt_data:
            qid = gt['question_id']
            if qid in pred_dict:
                matched.append((gt, pred_dict[qid]))
            else:
                print(f"⚠️ Warning: No prediction for {qid}")

        return matched

    # ============================================================
    # Task 1: Binary Classification (결손 유무)
    # ============================================================

    def eval_binary_classification(self) -> Dict:
        """Evaluate binary classification ONLY for *_missing_summary and *_missing_none"""
        print("\n" + "="*60)
        print("TASK 1: BINARY CLASSIFICATION (결손 유무)")
        print("Only for *_missing_summary and *_missing_none")
        print("="*60)

        y_true = []
        y_pred = []

        used = 0
        skipped = 0

        for gt, pred in self.matched_pairs:
            qid = gt.get("question_id", "")

            # ✅ Only evaluate summary / none questions
            if not (qid.endswith("_missing_summary") or qid.endswith("_missing_none")):
                skipped += 1
                continue

            used += 1

            # GT label
            gt_label = gt.get('label', gt.get('text', '')).strip().lower()
            gt_binary = 1 if gt_label in ['yes', 'y', 'true'] else 0

            # Predicted label
            pred_text = pred.get('text', '').strip().lower()
            if 'no missing' in pred_text or 'not missing' in pred_text:
                pred_binary = 0
            elif 'missing' in pred_text:
                pred_binary = 1
            else:
                pred_binary = 0

            y_true.append(gt_binary)
            y_pred.append(pred_binary)

        # Edge case: no samples
        if used == 0:
            print("⚠️ No *_missing_summary or *_missing_none samples found.")
            return {
                'accuracy': 0.0,
                'precision_no': 0.0, 'recall_no': 0.0, 'f1_no': 0.0, 'support_no': 0,
                'precision_yes': 0.0, 'recall_yes': 0.0, 'f1_yes': 0.0, 'support_yes': 0,
                'confusion_matrix': [[0, 0], [0, 0]],
                'used_samples': used,
                'skipped_samples': skipped,
                'filter_rule': "qid.endswith('_missing_summary') or qid.endswith('_missing_none')"
            }

        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1], zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        results = {
            'accuracy': float(accuracy),
            'precision_no': float(precision[0]),
            'recall_no': float(recall[0]),
            'f1_no': float(f1[0]),
            'support_no': int(support[0]),
            'precision_yes': float(precision[1]),
            'recall_yes': float(recall[1]),
            'f1_yes': float(f1[1]),
            'support_yes': int(support[1]),
            'confusion_matrix': cm.tolist(),
            'used_samples': used,
            'skipped_samples': skipped,
            'filter_rule': "qid.endswith('_missing_summary') or qid.endswith('_missing_none')"
        }

        # Print
        print(f"\n🧾 Filter rule:")
        print(f"   *_missing_summary, *_missing_none only")
        print(f"✅ Evaluated samples: {used}")
        print(f"⏭️  Skipped samples:   {skipped}")

        print(f"\n📊 Overall Accuracy: {accuracy*100:.2f}%")
        print(f"\n🔴 Class 'No Missing':")
        print(f"   Precision: {precision[0]*100:.2f}%")
        print(f"   Recall:    {recall[0]*100:.2f}%")
        print(f"   F1-Score:  {f1[0]*100:.2f}%")
        print(f"   Support:   {support[0]}")
        print(f"\n🟢 Class 'Yes Missing':")
        print(f"   Precision: {precision[1]*100:.2f}%")
        print(f"   Recall:    {recall[1]*100:.2f}%")
        print(f"   F1-Score:  {f1[1]*100:.2f}%")
        print(f"   Support:   {support[1]}")
        print(f"\n📈 Confusion Matrix:")
        print(f"   Predicted →    No   Yes")
        print(f"   Actual ↓")
        print(f"   No           {cm[0][0]:4d} {cm[0][1]:4d}")
        print(f"   Yes          {cm[1][0]:4d} {cm[1][1]:4d}")

        return results

    # ============================================================
    # Task 2: Component Type Classification (어떤 타입?)
    # ============================================================

    def extract_component_types(self, text: str) -> List[str]:
        """Extract mentioned component types from text"""
        text_lower = text.lower()
        types = []

        if 'vertical' in text_lower or 'post' in text_lower:
            types.append('vertical')
        if 'horizontal' in text_lower or 'beam' in text_lower:
            types.append('horizontal')
        if 'platform' in text_lower:
            types.append('platform')

        return types

    def infer_type_from_qid(self, qid: str) -> List[str]:
        """Infer component types from question_id pattern"""
        qid_lower = qid.lower()
        types = []
        if "_vertical_" in qid_lower:
            types.append("vertical")
        if "_horizontal_" in qid_lower:
            types.append("horizontal")
        if "_platform_" in qid_lower:
            types.append("platform")
        return types

    def eval_component_type(self) -> Dict:
        """Evaluate component type classification for questions with MISSING components (exclude positive)"""
        print("\n" + "="*60)
        print("TASK 2: COMPONENT TYPE CLASSIFICATION (어떤 타입?)")
        print("Exclude: *_missing_none, *_missing_positive_*")
        print("="*60)

        type_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

        used = 0
        skipped = 0

        for gt, pred in self.matched_pairs:
            qid = gt.get("question_id", "")

            # ✅ Exclude: none (no missing) and positive (existing components)
            if qid.endswith("_missing_none") or "_missing_positive_" in qid:
                skipped += 1
                continue

            # Only questions with bboxes (actual missing components)
            gt_bboxes = gt.get('bboxes', [])
            if not gt_bboxes:
                skipped += 1
                continue

            used += 1

            # Extract GT types from text
            gt_text = gt.get('text', '')
            gt_types = set(self.extract_component_types(gt_text))

            # ✅ qid 기반 보강 (text에서 못 뽑으면 qid에서라도 확보)
            if not gt_types:
                gt_types = set(self.infer_type_from_qid(qid))

            # Extract predicted types
            pred_text = pred.get('text', '')
            pred_types = set(self.extract_component_types(pred_text))

            # Calculate TP, FP, FN for each type
            for ctype in ['vertical', 'horizontal', 'platform']:
                if ctype in gt_types and ctype in pred_types:
                    type_metrics[ctype]['tp'] += 1
                elif ctype not in gt_types and ctype in pred_types:
                    type_metrics[ctype]['fp'] += 1
                elif ctype in gt_types and ctype not in pred_types:
                    type_metrics[ctype]['fn'] += 1

        # Calculate metrics
        results = {}
        f1_scores = []

        print(f"\n🧾 Filter rule:")
        print(f"   Exclude: *_missing_none, *_missing_positive_*")
        print(f"✅ Evaluated samples: {used}")
        print(f"⏭️  Skipped samples:   {skipped}")

        print()
        for ctype in ['vertical', 'horizontal', 'platform']:
            tp = type_metrics[ctype]['tp']
            fp = type_metrics[ctype]['fp']
            fn = type_metrics[ctype]['fn']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results[ctype] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'tp': tp, 'fp': fp, 'fn': fn
            }

            f1_scores.append(f1)

            print(f"📊 {ctype.capitalize()}:")
            print(f"   Precision: {precision*100:.2f}%")
            print(f"   Recall:    {recall*100:.2f}%")
            print(f"   F1-Score:  {f1*100:.2f}%")
            print(f"   (TP={tp}, FP={fp}, FN={fn})")

        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        results['macro_f1'] = float(macro_f1)
        results['used_samples'] = used
        results['skipped_samples'] = skipped
        results['filter_rule'] = "Exclude: *_missing_none, *_missing_positive_*"

        print(f"\n🎯 Macro-averaged F1: {macro_f1*100:.2f}%")

        return results

    # ============================================================
    # Task 3: BBox Grounding (어디서? - 3D 좌표)
    # ============================================================

    def extract_bboxes_from_text(self, text: str) -> List[List[List[float]]]:
        """Extract 8-corner bboxes from text using regex"""
        # Pattern: [[x,y,z], [x,y,z], ...]
        pattern = r'\[\s*\[\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*\](?:\s*,\s*\[\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*\])*\s*\]'

        matches = re.findall(pattern, text)

        bboxes = []
        for match in matches:
            try:
                bbox = ast.literal_eval(match)  # Safe parse
                if isinstance(bbox, list) and len(bbox) == 8:
                    # Validate it's 8 corners with 3D coordinates
                    if all(isinstance(corner, list) and len(corner) == 3 for corner in bbox):
                        bboxes.append(bbox)
            except:
                continue

        return bboxes

    def calculate_iou_3d(self, bbox1: List[List[float]], bbox2: List[List[float]]) -> float:
        """Calculate 3D IoU between two 8-corner bboxes"""
        try:
            # Convert to numpy arrays
            bbox1_array = np.array(bbox1, dtype=np.float32)
            bbox2_array = np.array(bbox2, dtype=np.float32)

            # Get min/max corners
            min1 = bbox1_array.min(axis=0)
            max1 = bbox1_array.max(axis=0)
            min2 = bbox2_array.min(axis=0)
            max2 = bbox2_array.max(axis=0)

            # Calculate intersection
            inter_min = np.maximum(min1, min2)
            inter_max = np.minimum(max1, max2)
            inter_dims = np.maximum(0, inter_max - inter_min)
            inter_volume = np.prod(inter_dims)

            # Calculate union
            vol1 = np.prod(max1 - min1)
            vol2 = np.prod(max2 - min2)
            union_volume = vol1 + vol2 - inter_volume

            # IoU
            iou = inter_volume / (union_volume + 1e-8)
            return float(iou)
        except:
            return 0.0

    def eval_bbox_grounding(self) -> Dict:
        """Evaluate 3D bounding box grounding for ALL questions with bboxes (include positive)"""
        print("\n" + "="*60)
        print("TASK 3: 3D BOUNDING BOX GROUNDING (어디서? - 3D 좌표)")
        print("Exclude: *_missing_none only")
        print("="*60)

        all_ious = []
        detected_count = 0
        total_gt_bboxes = 0

        iou_thresholds = [0.25, 0.5, 0.75]
        success_at_iou = {th: 0 for th in iou_thresholds}

        used = 0
        skipped = 0

        for gt, pred in self.matched_pairs:
            qid = gt.get("question_id", "")

            # ✅ Only exclude none (no bboxes)
            if qid.endswith("_missing_none"):
                skipped += 1
                continue

            gt_bboxes = gt.get('bboxes', [])
            if not gt_bboxes:
                skipped += 1
                continue

            used += 1
            total_gt_bboxes += len(gt_bboxes)

            # Extract predicted bboxes from text
            pred_text = pred.get('text', '')
            pred_bboxes = self.extract_bboxes_from_text(pred_text)

            if pred_bboxes:
                detected_count += 1

            # Match GT and Pred bboxes (greedy matching)
            for gt_bbox in gt_bboxes:
                if not pred_bboxes:
                    all_ious.append(0.0)
                    continue

                # Find best matching pred bbox
                best_iou = 0.0
                for pred_bbox in pred_bboxes:
                    iou = self.calculate_iou_3d(gt_bbox, pred_bbox)
                    best_iou = max(best_iou, iou)

                all_ious.append(best_iou)

                # Count for success rate@IoU
                for th in iou_thresholds:
                    if best_iou >= th:
                        success_at_iou[th] += 1

        # Calculate metrics
        mean_iou = np.mean(all_ious) if all_ious else 0.0
        median_iou = np.median(all_ious) if all_ious else 0.0
        detection_rate = detected_count / used if used > 0 else 0.0

        results = {
            'mean_iou': float(mean_iou),
            'median_iou': float(median_iou),
            'detection_rate': float(detection_rate),
            'total_gt_bboxes': total_gt_bboxes,
            'detected_samples': detected_count,
            'used_samples': used,
            'skipped_samples': skipped,
            'filter_rule': "Use samples with gt_bboxes only (skips *_missing_none and any gt_bboxes=[])"
        }

        for th in iou_thresholds:
            rate = success_at_iou[th] / total_gt_bboxes if total_gt_bboxes > 0 else 0.0
            results[f'success_rate@{th}'] = float(rate)

        # Print results
        print(f"\n🧾 Filter rule:")
        print(f"   Use only samples with GT bboxes (skip *_missing_none and any empty-bbox samples)")
        print(f"✅ Evaluated samples: {used}")
        print(f"⏭️  Skipped samples:   {skipped}")

        print(f"\n📊 BBox Detection Rate: {detection_rate*100:.2f}%")
        print(f"   ({detected_count}/{used} samples had predicted bboxes)")
        print(f"\n📏 IoU Statistics:")
        print(f"   Mean IoU:   {mean_iou:.4f}")
        print(f"   Median IoU: {median_iou:.4f}")
        print(f"\n🎯 Success Rate @ IoU Threshold (GT-based):")
        for th in iou_thresholds:
            rate = results[f'success_rate@{th}']
            print(f"   IoU ≥ {th}: {rate*100:.2f}%")

        return results

    # ============================================================
    # Task 4: Counting Accuracy (몇 개?)
    # ============================================================

    def extract_count(self, text: str) -> int:
        """Extract number of missing components from text"""
        # Pattern 1: "X components are missing" or "Missing: X"
        patterns = [
            r'(\d+)\s+components?\s+(?:are\s+)?missing',
            r'missing:\s*(\d+)',
            r'total:\s*(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))

        # Fallback: count bullet points with component descriptions
        count = text.count('- ') + text.count('• ')
        # Only count if there are actual component mentions
        if count > 0 and any(kw in text.lower() for kw in ['vertical', 'horizontal', 'platform']):
            return count

        return 0

    def eval_counting(self) -> Dict:
        """Evaluate counting accuracy for summary/floor/bay questions (exclude specific/positive - always 1)"""
        print("\n" + "="*60)
        print("TASK 4: COUNTING ACCURACY (몇 개?)")
        print("Only: *_summary, *_floor_*, *_bay_*")
        print("="*60)

        errors = []
        exact_matches = 0
        within_1 = 0

        used = 0
        skipped = 0

        for gt, pred in self.matched_pairs:
            qid = gt.get("question_id", "")

            # ✅ Only evaluate summary/floor/bay questions (multiple components)
            if not (qid.endswith("_missing_summary") or
                    "_missing_floor_" in qid or
                    "_missing_bay_" in qid or
                    qid.endswith("_missing_vertical_summary") or
                    qid.endswith("_missing_horizontal_summary")):
                skipped += 1
                continue

            used += 1

            # GT count from bboxes
            gt_count = len(gt.get('bboxes', []))

            # Predicted count from text
            pred_text = pred.get('text', '')
            pred_count = self.extract_count(pred_text)

            error = abs(gt_count - pred_count)
            errors.append(error)

            if error == 0:
                exact_matches += 1
            if error <= 1:
                within_1 += 1

        if used == 0:
            print("⚠️ No summary/floor/bay samples found.")
            return {
                'mae': 0.0,
                'exact_match_rate': 0.0,
                'within_1_rate': 0.0,
                'error_distribution': {},
                'used_samples': 0,
                'skipped_samples': skipped,
                'filter_rule': "Only: *_summary, *_floor_*, *_bay_*"
            }

        mae = np.mean(errors) if errors else 0.0
        exact_match_rate = exact_matches / len(errors) if errors else 0.0
        within_1_rate = within_1 / len(errors) if errors else 0.0

        # Error distribution
        error_dist = defaultdict(int)
        for e in errors:
            if e >= 3:
                error_dist['3+'] += 1
            else:
                error_dist[e] += 1

        results = {
            'mae': float(mae),
            'exact_match_rate': float(exact_match_rate),
            'within_1_rate': float(within_1_rate),
            'error_distribution': dict(error_dist),
            'used_samples': used,
            'skipped_samples': skipped,
            'filter_rule': "Only: *_summary, *_floor_*, *_bay_*"
        }

        print(f"\n🧾 Filter rule:")
        print(f"   Only: *_summary, *_floor_*, *_bay_*")
        print(f"✅ Evaluated samples: {used}")
        print(f"⏭️  Skipped samples:   {skipped}")

        print(f"\n📊 Mean Absolute Error (MAE): {mae:.3f}")
        print(f"📊 Exact Match Rate: {exact_match_rate*100:.2f}%")
        print(f"📊 Within ±1: {within_1_rate*100:.2f}%")
        print(f"\n📈 Error Distribution:")
        total = len(errors)
        for error_val in sorted([k for k in error_dist.keys() if k != '3+']):
            count = error_dist[error_val]
            print(f"   Error={error_val}: {count:4d} ({count/total*100:5.1f}%)")
        if '3+' in error_dist:
            count = error_dist['3+']
            print(f"   Error≥3:  {count:4d} ({count/total*100:5.1f}%)")

        return results

    # ============================================================
    # Task 5: Spatial Reasoning (어디서? - 층/열)
    # ============================================================

    def extract_question_type(self, question_id: str) -> Tuple[str, Optional[int]]:
        """Extract question type and target (floor/bay) from question_id"""
        if '_floor_' in question_id:
            match = re.search(r'_floor_(\d+)', question_id)
            floor = int(match.group(1)) if match else None
            return ('floor', floor)
        elif '_bay_' in question_id:
            match = re.search(r'_bay_(\d+)', question_id)
            bay = int(match.group(1)) if match else None
            return ('bay', bay)
        else:
            return ('general', None)

    def eval_spatial_reasoning(self) -> Dict:
        """Evaluate spatial reasoning ONLY for *_missing_floor_* and *_missing_bay_*"""
        print("\n" + "="*60)
        print("TASK 5: SPATIAL REASONING (어디서? - 층/열)")
        print("Only: *_missing_floor_*, *_missing_bay_*")
        print("="*60)

        floor_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        bay_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})

        used = 0
        skipped = 0

        for gt, pred in self.matched_pairs:
            qid = gt['question_id']

            qtype, target = self.extract_question_type(qid)

            if qtype not in ['floor', 'bay']:
                skipped += 1
                continue

            used += 1

            pred_text = pred.get('text', '').lower()

            # ✅ GT truth: bboxes 기반 (label 사용 금지)
            gt_has_missing = len(gt.get('bboxes', [])) > 0

            # ✅ Pred judgement
            pred_says_no_missing = ('no missing' in pred_text) or ('not missing' in pred_text) or ('none' in pred_text)
            pred_says_missing = ('missing' in pred_text) and (not pred_says_no_missing)

            # ✅ 해당 floor/bay 언급 여부로 "공간"성 강화 (유연한 표현 허용)
            mentions_target = True
            if target is not None:
                if qtype == 'floor':
                    # Allow 'floor' or 'level' keywords
                    mentions_target = (str(target) in pred_text) and any(k in pred_text for k in ['floor', 'level'])
                elif qtype == 'bay':
                    mentions_target = (str(target) in pred_text) and ('bay' in pred_text)

            # ✅ Correctness
            if gt_has_missing:
                correct = pred_says_missing and mentions_target
            else:
                correct = pred_says_no_missing and mentions_target

            if qtype == 'floor':
                floor_metrics[target]['correct'] += int(correct)
                floor_metrics[target]['total'] += 1
            elif qtype == 'bay':
                bay_metrics[target]['correct'] += int(correct)
                bay_metrics[target]['total'] += 1

        results = {'floor': {}, 'bay': {}}

        print(f"\n🧾 Filter rule:")
        print(f"   Only: *_missing_floor_*, *_missing_bay_*")
        print(f"✅ Evaluated samples: {used}")
        print(f"⏭️  Skipped samples:   {skipped}")

        print("\n📊 Floor-specific Questions:")
        for floor in sorted(floor_metrics.keys()):
            metrics = floor_metrics[floor]
            acc = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0.0
            results['floor'][floor] = float(acc)
            print(f"   Floor {floor}: {acc*100:.2f}% ({metrics['correct']}/{metrics['total']})")

        print("\n📊 Bay-specific Questions:")
        for bay in sorted(bay_metrics.keys()):
            metrics = bay_metrics[bay]
            acc = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0.0
            results['bay'][bay] = float(acc)
            print(f"   Bay {bay}: {acc*100:.2f}% ({metrics['correct']}/{metrics['total']})")

        # Overall spatial accuracy
        total_correct = sum(m['correct'] for m in floor_metrics.values()) + \
                       sum(m['correct'] for m in bay_metrics.values())
        total_count = sum(m['total'] for m in floor_metrics.values()) + \
                     sum(m['total'] for m in bay_metrics.values())

        overall_acc = total_correct / total_count if total_count > 0 else 0.0
        results['overall_accuracy'] = float(overall_acc)
        results['used_samples'] = used
        results['skipped_samples'] = skipped
        results['filter_rule'] = "Only: *_missing_floor_*, *_missing_bay_*"

        print(f"\n🎯 Overall Spatial Accuracy: {overall_acc*100:.2f}%")

        return results

    # ============================================================
    # Task 6: Template-Guided Format Validation
    # ============================================================

    def validate_template_format(self, text: str) -> Dict[str, bool]:
        """Check if answer follows Expected vs Actual template"""
        text_lower = text.lower()

        has_expected = 'expected' in text_lower
        has_actual = 'actual' in text_lower
        has_structure = any(kw in text_lower for kw in ['bay', 'row', 'floor', 'scaffold'])
        has_bbox = '[[' in text and ']]' in text

        format_correct = has_expected and has_actual

        return {
            'has_expected': has_expected,
            'has_actual': has_actual,
            'has_structure': has_structure,
            'has_bbox': has_bbox,
            'format_correct': format_correct
        }

    def eval_template_format(self) -> Dict:
        """Evaluate template format for summary questions (Expected/Actual format)"""
        print("\n" + "="*60)
        print("TASK 6: TEMPLATE-GUIDED FORMAT VALIDATION")
        print("Only: *_summary (Expected/Actual format)")
        print("="*60)

        format_stats = {
            'has_expected': 0,
            'has_actual': 0,
            'has_structure': 0,
            'has_bbox': 0,
            'format_correct': 0
        }

        format_errors = []

        used = 0
        skipped = 0

        for gt, pred in self.matched_pairs:
            qid = gt.get("question_id", "")

            # ✅ Only evaluate summary questions (they have Expected/Actual template)
            if not (qid.endswith("_missing_summary") or
                    qid.endswith("_missing_none") or
                    qid.endswith("_missing_vertical_summary") or
                    qid.endswith("_missing_horizontal_summary")):
                skipped += 1
                continue

            used += 1

            pred_text = pred.get('text', '')
            validation = self.validate_template_format(pred_text)

            for key in format_stats.keys():
                if validation[key]:
                    format_stats[key] += 1

            if not validation['format_correct']:
                # Only check required fields for missing
                required = ['has_expected', 'has_actual']
                missing = [k for k in required if not validation[k]]
                format_errors.append({
                    'question_id': pred['question_id'],
                    'missing': missing
                })

        total = used
        results = {k: v/total if total > 0 else 0.0 for k, v in format_stats.items()}
        results['total_samples'] = total
        results['error_count'] = len(format_errors)
        results['used_samples'] = used
        results['skipped_samples'] = skipped
        results['filter_rule'] = "Only: *_summary (Expected/Actual template)"

        print(f"\n🧾 Filter rule:")
        print(f"   Only: *_summary questions (Expected/Actual template)")
        print(f"✅ Evaluated samples: {used}")
        print(f"⏭️  Skipped samples:   {skipped}")

        print(f"\n📊 Format Compliance:")
        print(f"   Has 'Expected': {format_stats['has_expected']}/{total} ({results['has_expected']*100:.1f}%)")
        print(f"   Has 'Actual':   {format_stats['has_actual']}/{total} ({results['has_actual']*100:.1f}%)")
        print(f"   Has structure:  {format_stats['has_structure']}/{total} ({results['has_structure']*100:.1f}%)")
        print(f"   Has BBox:       {format_stats['has_bbox']}/{total} ({results['has_bbox']*100:.1f}%)")
        print(f"\n🎯 Full Format Compliance: {format_stats['format_correct']}/{total} ({results['format_correct']*100:.1f}%)")

        if format_errors:
            print(f"\n⚠️ Format errors found: {len(format_errors)} cases")
            print("   (See detailed report in JSON output)")

        return results

    # ============================================================
    # Comprehensive Evaluation
    # ============================================================

    def evaluate_all(self) -> Dict:
        """Run all evaluation tasks"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION FOR SCAFFOLD MISSING DETECTION")
        print("Question-type filtered based on actual data generation patterns")
        print("="*80)
        print(f"Dataset: {len(self.matched_pairs)} matched samples")

        results = {}

        # Run all tasks
        results['task1_binary_classification'] = self.eval_binary_classification()
        results['task2_component_type'] = self.eval_component_type()
        results['task3_bbox_grounding'] = self.eval_bbox_grounding()
        results['task4_counting'] = self.eval_counting()
        results['task5_spatial_reasoning'] = self.eval_spatial_reasoning()
        results['task6_template_format'] = self.eval_template_format()

        # Summary
        self.print_summary(results)

        return results

    def print_summary(self, results: Dict):
        """Print summary of all metrics"""
        print("\n" + "="*80)
        print("SUMMARY METRICS")
        print("="*80)

        print(f"\n✅ Task 1 - Binary Classification (결손 유무):")
        print(f"   Accuracy: {results['task1_binary_classification']['accuracy']*100:.2f}%")
        print(f"   Samples:  {results['task1_binary_classification']['used_samples']}")

        print(f"\n✅ Task 2 - Component Type (어떤 타입?):")
        print(f"   Macro F1: {results['task2_component_type']['macro_f1']*100:.2f}%")
        print(f"   Samples:  {results['task2_component_type']['used_samples']}")

        print(f"\n🎯 Task 3 - BBox Grounding (어디서? - 3D 좌표):")
        print(f"   Mean IoU:          {results['task3_bbox_grounding']['mean_iou']:.4f}")
        print(f"   Success Rate@0.5:  {results['task3_bbox_grounding']['success_rate@0.5']*100:.2f}%")
        print(f"   Samples:           {results['task3_bbox_grounding']['used_samples']}")

        print(f"\n✅ Task 4 - Counting (몇 개?):")
        print(f"   MAE:          {results['task4_counting']['mae']:.3f}")
        print(f"   Exact Match:  {results['task4_counting']['exact_match_rate']*100:.2f}%")
        print(f"   Samples:      {results['task4_counting']['used_samples']}")

        print(f"\n✅ Task 5 - Spatial Reasoning (어디서? - 층/열):")
        print(f"   Overall: {results['task5_spatial_reasoning']['overall_accuracy']*100:.2f}%")
        print(f"   Samples: {results['task5_spatial_reasoning']['used_samples']}")

        print(f"\n✅ Task 6 - Template Format:")
        print(f"   Compliance: {results['task6_template_format']['format_correct']*100:.2f}%")
        print(f"   Samples:    {results['task6_template_format']['used_samples']}")

        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation for Scaffold Missing Detection')
    parser.add_argument('--gt-file', type=str, required=True,
                       help='Ground truth JSONL file (gt_missing.jsonl)')
    parser.add_argument('--pred-file', type=str, required=True,
                       help='Prediction JSONL file (answers_missing.jsonl)')
    parser.add_argument('--output-json', type=str, default='eval_results.json',
                       help='Output JSON file for detailed results')
    parser.add_argument('--output-report', type=str, default='eval_report.txt',
                       help='Output text report')

    args = parser.parse_args()

    # Run evaluation
    evaluator = ComprehensiveMissingEvaluator(args.gt_file, args.pred_file)
    results = evaluator.evaluate_all()

    # Save results
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {args.output_json}")
    print(f"✅ Evaluation complete!")


if __name__ == '__main__':
    main()