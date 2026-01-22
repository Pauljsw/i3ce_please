"""
Enhanced evaluation script for scaffold missing component detection.

This script provides more robust answer parsing compared to the original
eval_missing_detection.py, handling various response formats.
"""

import argparse
import json
import re
from collections import Counter

def parse_answer(pred_text: str) -> str:
    pred_text = pred_text.strip().lower()
    
    # 1. 명확한 yes/no가 있으면 우선 적용
    has_yes = re.search(r'\byes\b', pred_text)
    has_no = re.search(r'\bno\b', pred_text)
    
    if has_yes and not has_no:
        return 'yes'
    elif has_no and not has_yes:
        return 'no'
    elif has_yes and has_no:
        # 둘 다 있으면 문맥으로 판단
        # "Yes, there are missing..." vs "No, there are no missing..."
        if re.search(r'no.*missing', pred_text):
            return 'no'
        elif re.search(r'yes.*missing', pred_text):
            return 'yes'
    
    # 2. Yes/No 없으면 패턴 매칭
    # 부정 패턴을 먼저 체크 (더 구체적이므로)
    negative_patterns = [
        r'no missing',
        r'not missing', 
        r'no.*(?:components?|platforms?|verticals?|horizontals?).*missing',
        r'all.*(?:components?|platforms?).*(?:are|is).*(?:present|installed)',
    ]
    
    for pattern in negative_patterns:
        if re.search(pattern, pred_text):
            return 'no'
    
    # 긍정 패턴 (부정이 없는 경우만)
    positive_patterns = [
        r'(?:components?|platforms?|verticals?|horizontals?).*(?:are|is).*missing',
        r'missing.*(?:components?|platforms?|verticals?|horizontals?)',
        r'absent.*(?:components?|platforms?)',
    ]
    
    for pattern in positive_patterns:
        if re.search(pattern, pred_text):
            return 'yes'
    
    return 'unknown'


def evaluate_missing_detection(answers_file: str, gt_file: str) -> None:
    """Evaluate missing detection predictions with enhanced parsing."""
    
    # Load ground truth labels
    gt_labels = {}
    gt_categories = Counter()
    
    with open(gt_file, 'r', encoding='utf-8') as f_gt:
        for line in f_gt:
            record = json.loads(line)
            qid = record['question_id']
            label = record['text'].strip().lower()
            gt_labels[qid] = label
            
            # Extract question type for detailed analysis
            if '_summary' in qid:
                gt_categories['summary'] += 1
            elif '_vertical_' in qid:
                gt_categories['vertical'] += 1
            elif '_horizontal_' in qid:
                gt_categories['horizontal'] += 1
            elif '_floor_' in qid:
                gt_categories['floor'] += 1
            elif '_bay_' in qid:
                gt_categories['bay'] += 1
            elif '_positive_' in qid:
                gt_categories['positive'] += 1
            elif '_none' in qid:
                gt_categories['none'] += 1
            else:
                gt_categories['other'] += 1

    # Evaluate predictions
    total = 0
    correct = 0
    confusion = Counter()
    category_results = {}
    unknown_predictions = []

    with open(answers_file, 'r', encoding='utf-8') as f_ans:
        for line in f_ans:
            ans = json.loads(line)
            qid = ans.get('question_id')
            pred_text = ans.get('text', '').strip()
            
            # Enhanced answer parsing
            pred_label = parse_answer(pred_text)
            
            # Track unknown predictions for analysis
            if pred_label == 'unknown':
                unknown_predictions.append({
                    'question_id': qid,
                    'prediction': pred_text[:100]  # First 100 chars
                })
            
            # Compare with ground truth
            gt_label = gt_labels.get(qid)
            if gt_label is None:
                continue
                
            total += 1
            confusion[(gt_label, pred_label)] += 1
            
            # Category-wise accuracy
            category = None
            if '_summary' in qid:
                category = 'summary'
            elif '_vertical_' in qid:
                category = 'vertical'
            elif '_horizontal_' in qid:
                category = 'horizontal'
            elif '_floor_' in qid:
                category = 'floor'
            elif '_bay_' in qid:
                category = 'bay'
            elif '_positive_' in qid:
                category = 'positive'
            elif '_none' in qid:
                category = 'none'
            else:
                category = 'other'
                
            if category not in category_results:
                category_results[category] = {'total': 0, 'correct': 0}
            category_results[category]['total'] += 1
            
            if pred_label == gt_label:
                correct += 1
                category_results[category]['correct'] += 1

    # Print results
    overall_accuracy = correct / total if total > 0 else 0.0
    
    print("=" * 60)
    print("🎯 MISSING DETECTION EVALUATION RESULTS")
    print("=" * 60)
    print(f"📊 Overall Statistics:")
    print(f"   Total questions: {total}")
    print(f"   Correct predictions: {correct}")
    print(f"   Overall accuracy: {overall_accuracy:.4f}")
    print(f"   Unknown predictions: {len(unknown_predictions)}")
    
    print(f"\n📈 Category-wise Accuracy:")
    for category, results in sorted(category_results.items()):
        if results['total'] > 0:
            cat_acc = results['correct'] / results['total']
            print(f"   {category:>12}: {cat_acc:.4f} ({results['correct']}/{results['total']})")
    
    print(f"\n🔍 Confusion Matrix (ground_truth, prediction):")
    for (gt, pred), count in sorted(confusion.items()):
        percentage = count / total * 100
        print(f"   ({gt:>3}, {pred:>7}): {count:>4} ({percentage:5.1f}%)")
    
    # Calculate precision, recall, F1 for "yes" class
    tp = confusion[('yes', 'yes')]
    fp = confusion[('no', 'yes')] + confusion[('yes', 'unknown')]
    fn = confusion[('yes', 'no')] + confusion[('yes', 'unknown')]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📏 Detection Metrics (for 'Yes' class):")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    # Show unknown predictions for debugging
    if unknown_predictions:
        print(f"\n⚠️  Unknown Predictions (sample):")
        for item in unknown_predictions[:5]:  # Show first 5
            print(f"   {item['question_id']}: {item['prediction']}")
        if len(unknown_predictions) > 5:
            print(f"   ... and {len(unknown_predictions) - 5} more")
    
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced missing detection evaluation.')
    parser.add_argument('--answers-file', type=str, required=True,
                        help='Path to model prediction JSONL file.')
    parser.add_argument('--gt-file', type=str, required=True,
                        help='Path to ground truth JSONL file.')
    args = parser.parse_args()
    evaluate_missing_detection(args.answers_file, args.gt_file)