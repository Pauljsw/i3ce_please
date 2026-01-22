#!/usr/bin/env python3
"""
GPT-based Evaluation for Scaffold Missing Detection

This module provides GPT-4/3.5 based semantic evaluation for:
1. Answer Quality - Overall semantic similarity
2. Component Detection - Correct component type identification
3. Location Accuracy - Whether described locations make sense
4. Count Verification - Number of missing components

Usage:
    python tools/gpt_evaluate_scaffold.py \
        --predictions ./outputs/test_answers.jsonl \
        --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
        --questions ./playground/data/shapellm/scaffold_v2/test_questions.jsonl \
        --output-dir ./evaluation_results \
        --openai-key YOUR_API_KEY \
        --model gpt-4

Data Format Expected:
    - predictions: {question_id, text: <model_output>, ...}
    - ground-truth: {question_id, text: <gt_answer>, label, task_type, ...}
    - questions: {question_id, text: <question_text>, category, ...}

Based on ShapeLLM's gpt_eval.py and eval_3dmmvet.py
"""

import os
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# OpenAI API
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai package not installed. Install with: pip install openai")


# ============================================================================
# Configuration
# ============================================================================

EVALUATION_CATEGORIES = [
    "Binary Detection",      # Yes/No 판단 정확도
    "Component Type",        # 부재 종류 식별
    "Count Accuracy",        # 개수 정확도
    "Location Description",  # 위치 설명 정확도
    "Overall Quality",       # 전체 답변 품질
]

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2.0


# ============================================================================
# GPT Evaluation Prompts
# ============================================================================

SYSTEM_PROMPT = """You are an expert evaluator for 3D scaffold inspection tasks.
Your job is to evaluate how well a model's answer matches the ground truth answer
for scaffold missing component detection questions.

You must be strict but fair:
- Focus on the correctness of the information, not the exact wording
- A correct detection with slightly different phrasing should still score high
- Missing critical information (like missing components not mentioned) should score low
- Wrong component types or counts should score low
"""

EVALUATION_PROMPT_TEMPLATE = """Evaluate the model's answer against the ground truth for this scaffold inspection question.

**Question:** {question}

**Ground Truth Answer:** {gt_answer}

**Model's Answer:** {pred_answer}

Evaluate on these criteria (0-100 each):

1. **Binary Detection** (0 or 100 ONLY): Did the model correctly identify if there are missing components?
   - 100: Correct Yes/No answer (model's answer matches ground truth on presence/absence)
   - 0: Wrong Yes/No answer (model said Yes when GT says No, or vice versa)
   - IMPORTANT: Only use 0 or 100 for this category, no intermediate values

2. **Component Type** (0-100): Did the model correctly identify the types of missing components?
   - 100: All component types correct (vertical/horizontal/platform)
   - 50: Some types correct
   - 0: All types wrong or not mentioned when should be

3. **Count Accuracy** (0-100): Did the model give the correct count of missing components?
   - 100: Exact count match
   - 70: Off by 1
   - 40: Off by 2
   - 0: Off by more than 2 or completely wrong

4. **Location Description** (0-100): How accurate are the location descriptions?
   - 100: Locations match ground truth
   - 50: Partially correct locations
   - 0: Wrong or missing locations

5. **Overall Quality** (0-100): Overall answer quality considering all factors
   - Consider completeness, accuracy, and usefulness

Respond ONLY with a JSON object in this exact format:
{{"binary_detection": <score>, "component_type": <score>, "count_accuracy": <score>, "location_description": <score>, "overall_quality": <score>}}
"""

SIMPLE_SCORE_PROMPT = """Compare these two answers for a scaffold inspection question and give a similarity score from 0-100.

Question: {question}
Ground Truth: {gt_answer}
Model Answer: {pred_answer}

Scoring guide:
- 100: Semantically identical, all key information matches
- 80-99: Correct answer with minor differences in wording
- 60-79: Mostly correct but missing some details
- 40-59: Partially correct with significant omissions or errors
- 20-39: Mostly incorrect
- 0-19: Completely wrong

Respond with ONLY a number between 0 and 100, nothing else."""


# ============================================================================
# GPT API Functions
# ============================================================================

def call_gpt_api(
    messages: List[Dict],
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.1,
    max_retries: int = MAX_RETRIES
) -> Optional[str]:
    """Call GPT API with retry logic."""
    if not HAS_OPENAI:
        return None

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=200
            )
            return response.choices[0].message['content'].strip()
        except openai.error.RateLimitError:
            time.sleep(RETRY_DELAY * (attempt + 1))
        except openai.error.APIError as e:
            print(f"API Error: {e}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    return None


def get_detailed_scores(
    question: str,
    gt_answer: str,
    pred_answer: str,
    model: str = "gpt-3.5-turbo"
) -> Optional[Dict[str, float]]:
    """Get detailed category scores from GPT."""

    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        question=question,
        gt_answer=gt_answer,
        pred_answer=pred_answer
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    response = call_gpt_api(messages, model=model)

    if not response:
        return None

    # Parse JSON response
    try:
        # Clean up response
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]

        scores = json.loads(response)

        # Validate scores
        valid_scores = {}
        for key in ["binary_detection", "component_type", "count_accuracy",
                    "location_description", "overall_quality"]:
            if key in scores:
                score = float(scores[key])
                valid_scores[key] = max(0, min(100, score))
            else:
                valid_scores[key] = 0.0

        return valid_scores

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse GPT response: {response[:100]}...")
        return None


def get_simple_score(
    question: str,
    gt_answer: str,
    pred_answer: str,
    model: str = "gpt-3.5-turbo"
) -> float:
    """Get simple 0-100 similarity score."""

    prompt = SIMPLE_SCORE_PROMPT.format(
        question=question,
        gt_answer=gt_answer,
        pred_answer=pred_answer
    )

    messages = [
        {"role": "system", "content": "You are an evaluation assistant. Respond only with a number."},
        {"role": "user", "content": prompt}
    ]

    response = call_gpt_api(messages, model=model)

    if not response:
        return -1.0

    try:
        # Extract number from response
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        if numbers:
            score = float(numbers[0])
            return max(0, min(100, score))
    except ValueError:
        pass

    return -1.0


def get_average_score(
    question: str,
    gt_answer: str,
    pred_answer: str,
    model: str = "gpt-3.5-turbo",
    times: int = 3
) -> float:
    """Get average score over multiple runs for stability."""
    scores = []

    for _ in range(times):
        score = get_simple_score(question, gt_answer, pred_answer, model)
        if score >= 0:
            scores.append(score)

    if not scores:
        return -1.0

    return sum(scores) / len(scores)


# ============================================================================
# Batch Evaluation
# ============================================================================

@dataclass
class GPTEvalResult:
    """Result of GPT evaluation for one sample."""
    question_id: str
    question: str
    gt_answer: str
    pred_answer: str
    scores: Dict[str, float]
    simple_score: float


@dataclass
class GPTEvalSummary:
    """Summary of GPT evaluation."""
    num_samples: int
    num_evaluated: int
    num_failed: int

    # Category averages
    binary_detection_avg: float
    component_type_avg: float
    count_accuracy_avg: float
    location_description_avg: float
    overall_quality_avg: float

    # Simple score average
    simple_score_avg: float

    # Per-task scores
    per_task_scores: Dict[str, float]


def evaluate_batch_parallel(
    samples: List[Dict],
    model: str = "gpt-3.5-turbo",
    max_workers: int = 4,
    detailed: bool = True
) -> Tuple[List[GPTEvalResult], GPTEvalSummary]:
    """Evaluate samples in parallel using thread pool."""

    results = []
    failed = 0

    # Category accumulators
    category_scores = defaultdict(list)
    simple_scores = []
    task_scores = defaultdict(list)

    def evaluate_one(sample: Dict) -> Optional[GPTEvalResult]:
        qid = sample.get('question_id', '')
        question = sample.get('question', '')
        gt_answer = sample.get('gt_answer', '')
        pred_answer = sample.get('pred_answer', '')
        task_type = sample.get('task_type', 'unknown')

        if detailed:
            scores = get_detailed_scores(question, gt_answer, pred_answer, model)
            simple = scores.get('overall_quality', -1) if scores else -1
        else:
            scores = {}
            simple = get_simple_score(question, gt_answer, pred_answer, model)

        if scores is None and simple < 0:
            return None

        return GPTEvalResult(
            question_id=qid,
            question=question,
            gt_answer=gt_answer,
            pred_answer=pred_answer,
            scores=scores or {},
            simple_score=simple
        ), task_type

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_one, s): s for s in samples}

        for future in tqdm(as_completed(futures), total=len(samples), desc="GPT Evaluation"):
            try:
                result = future.result()
                if result is None:
                    failed += 1
                    continue

                eval_result, task_type = result
                results.append(eval_result)

                # Accumulate scores
                for key, value in eval_result.scores.items():
                    category_scores[key].append(value)

                if eval_result.simple_score >= 0:
                    simple_scores.append(eval_result.simple_score)
                    task_scores[task_type].append(eval_result.simple_score)

            except Exception as e:
                print(f"Evaluation error: {e}")
                failed += 1

    # Calculate summary
    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    summary = GPTEvalSummary(
        num_samples=len(samples),
        num_evaluated=len(results),
        num_failed=failed,
        binary_detection_avg=safe_mean(category_scores.get('binary_detection', [])),
        component_type_avg=safe_mean(category_scores.get('component_type', [])),
        count_accuracy_avg=safe_mean(category_scores.get('count_accuracy', [])),
        location_description_avg=safe_mean(category_scores.get('location_description', [])),
        overall_quality_avg=safe_mean(category_scores.get('overall_quality', [])),
        simple_score_avg=safe_mean(simple_scores),
        per_task_scores={task: safe_mean(scores) for task, scores in task_scores.items()}
    )

    return results, summary


# ============================================================================
# Main Evaluation Function
# ============================================================================

def run_gpt_evaluation(
    predictions_path: str,
    ground_truth_path: str,
    questions_path: str,
    output_dir: str,
    api_key: str,
    model: str = "gpt-3.5-turbo",
    max_samples: int = None,
    detailed: bool = True,
    max_workers: int = 4
) -> GPTEvalSummary:
    """Run full GPT evaluation pipeline.

    Args:
        predictions_path: Path to model predictions JSONL (question_id, text=model_output)
        ground_truth_path: Path to ground truth JSONL (question_id, text=gt_answer, label, task_type)
        questions_path: Path to questions JSONL (question_id, text=question_text, category)
        output_dir: Directory for output files
        api_key: OpenAI API key
        model: GPT model to use
        max_samples: Limit number of samples (for testing)
        detailed: Use detailed 5-category scoring vs simple scoring
        max_workers: Parallel workers for API calls
    """

    if not HAS_OPENAI:
        raise ImportError("openai package is required. Install with: pip install openai")

    openai.api_key = api_key

    # Load data - predictions
    print(f"Loading predictions from: {predictions_path}")
    with open(predictions_path, 'r') as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    # Load data - ground truth (contains answer text, label, task_type)
    print(f"Loading ground truth from: {ground_truth_path}")
    with open(ground_truth_path, 'r') as f:
        ground_truth = [json.loads(line) for line in f if line.strip()]

    # Load data - questions (contains question text)
    print(f"Loading questions from: {questions_path}")
    with open(questions_path, 'r') as f:
        questions = [json.loads(line) for line in f if line.strip()]

    # Build lookups by question_id
    gt_lookup = {gt['question_id']: gt for gt in ground_truth}
    question_lookup = {q['question_id']: q for q in questions}

    print(f"  Predictions: {len(predictions)}")
    print(f"  Ground truth: {len(ground_truth)}")
    print(f"  Questions: {len(questions)}")

    # Prepare samples for evaluation
    samples = []
    missing_count = 0
    for pred in predictions:
        qid = pred.get('question_id')

        # Check both GT and question exist
        if qid not in gt_lookup:
            missing_count += 1
            continue
        if qid not in question_lookup:
            missing_count += 1
            continue

        gt = gt_lookup[qid]
        q = question_lookup[qid]

        samples.append({
            'question_id': qid,
            'question': q.get('text', ''),        # Question text from questions file
            'gt_answer': gt.get('text', ''),      # GT answer from ground truth file
            'pred_answer': pred.get('text', ''),  # Model output from predictions file
            'task_type': gt.get('task_type', 'unknown')
        })

    if missing_count > 0:
        print(f"  Warning: {missing_count} predictions missing from GT/questions")

    if max_samples:
        samples = samples[:max_samples]

    print(f"Evaluating {len(samples)} samples with {model}...")

    # Run evaluation
    results, summary = evaluate_batch_parallel(
        samples,
        model=model,
        max_workers=max_workers,
        detailed=detailed
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Individual results
    results_path = os.path.join(output_dir, 'gpt_eval_results.jsonl')
    with open(results_path, 'w') as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + '\n')

    # Summary
    summary_path = os.path.join(output_dir, 'gpt_eval_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(asdict(summary), f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print(" GPT EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nSamples: {summary.num_evaluated}/{summary.num_samples} evaluated, {summary.num_failed} failed")

    if detailed:
        print("\n[Category Scores]")
        print(f"  Binary Detection:     {summary.binary_detection_avg:.1f}/100")
        print(f"  Component Type:       {summary.component_type_avg:.1f}/100")
        print(f"  Count Accuracy:       {summary.count_accuracy_avg:.1f}/100")
        print(f"  Location Description: {summary.location_description_avg:.1f}/100")
        print(f"  Overall Quality:      {summary.overall_quality_avg:.1f}/100")

    print(f"\n[Simple Score Average]: {summary.simple_score_avg:.1f}/100")

    if summary.per_task_scores:
        print("\n[Per-Task Scores]")
        for task, score in sorted(summary.per_task_scores.items()):
            print(f"  {task}: {score:.1f}/100")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}")

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GPT-based Scaffold Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python tools/gpt_evaluate_scaffold.py \\
        --predictions ./outputs/test_answers.jsonl \\
        --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \\
        --questions ./playground/data/shapellm/scaffold_v2/test_questions.jsonl \\
        --output-dir ./evaluation_results \\
        --openai-key YOUR_API_KEY

Data format:
    - predictions: {"question_id": "...", "text": "<model_output>", ...}
    - ground-truth: {"question_id": "...", "text": "<gt_answer>", "label": "Yes/No", "task_type": "...", ...}
    - questions: {"question_id": "...", "text": "<question_text>", "category": "scaffold", ...}
        """
    )
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSONL (model outputs)')
    parser.add_argument('--ground-truth', type=str, required=True,
                       help='Path to ground truth JSONL (GT answers)')
    parser.add_argument('--questions', type=str, required=True,
                       help='Path to questions JSONL (question text)')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory')
    parser.add_argument('--openai-key', type=str, required=True,
                       help='OpenAI API key')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'],
                       help='GPT model to use')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to evaluate (for testing)')
    parser.add_argument('--simple', action='store_true',
                       help='Use simple scoring instead of detailed')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of parallel workers')

    args = parser.parse_args()

    run_gpt_evaluation(
        predictions_path=args.predictions,
        ground_truth_path=args.ground_truth,
        questions_path=args.questions,
        output_dir=args.output_dir,
        api_key=args.openai_key,
        model=args.model,
        max_samples=args.max_samples,
        detailed=not args.simple,
        max_workers=args.max_workers
    )


if __name__ == '__main__':
    main()
