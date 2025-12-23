"""
Scaffold dataset validation script
Based on llava/eval/eval_objaverse.py with modifications for scaffold-specific evaluation
"""

import argparse
from tqdm import tqdm
import torch
from transformers import StoppingCriteria
from torch.utils.data import Dataset, DataLoader
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_point_token, get_model_name_from_path, load_pts
from llava.constants import POINT_TOKEN_INDEX, DEFAULT_POINT_TOKEN, DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN

import os
import json
import numpy as np


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if
                            type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


class ScaffoldValidationDataset(Dataset):
    """Dataset for scaffold validation"""
    
    def __init__(self, data_path, anno_path, pointnum=10000, use_color=True):
        self.data_path = data_path
        self.pointnum = pointnum
        self.use_color = use_color
        
        # Load annotations
        with open(anno_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        print(f"Loaded {len(self.annotations)} conversation samples")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        anno = self.annotations[idx]
        
        # Load point cloud
        point_file = anno['point']
        point_path = os.path.join(self.data_path, point_file)
        
        # Load .npy file
        point_cloud = np.load(point_path)
        
        # Sample points if needed
        if point_cloud.shape[0] > self.pointnum:
            indices = np.random.choice(point_cloud.shape[0], self.pointnum, replace=False)
            point_cloud = point_cloud[indices]
        elif point_cloud.shape[0] < self.pointnum:
            # Upsample if not enough points
            indices = np.random.choice(point_cloud.shape[0], self.pointnum, replace=True)
            point_cloud = point_cloud[indices]
        
        # Convert to tensor
        point_cloud = torch.from_numpy(point_cloud.astype(np.float32))
        
        return {
            'point_cloud': point_cloud,
            'annotation': anno,
            'idx': idx
        }


def init_model(args):
    """Initialize model and tokenizer"""
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    conv_mode = "vicuna_v1"
    conv = conv_templates[conv_mode].copy()
    return model, tokenizer, conv


def collate_fn(batch):
    """Custom collate function"""
    point_clouds = torch.stack([item['point_cloud'] for item in batch])
    annotations = [item['annotation'] for item in batch]
    indices = [item['idx'] for item in batch]
    
    return {
        'point_clouds': point_clouds,
        'annotations': annotations,
        'indices': indices
    }


def evaluate_scaffold(model, tokenizer, conv, dataloader, args):
    """Run inference and evaluation on scaffold dataset"""
    results = []
    
    print("\n" + "="*80)
    print("Starting validation inference...")
    print("="*80 + "\n")
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            point_clouds = batch['point_clouds'].cuda()
            annotations = batch['annotations']
            
            # Process each sample in batch
            for idx in range(len(annotations)):
                anno = annotations[idx]
                point_cloud = point_clouds[idx].unsqueeze(0)
                
                # Get conversation
                conversations = anno['conversations']
                question = conversations[0]['value']
                ground_truth = conversations[1]['value']
                
                # Build conversation
                conv_instance = conv.copy()
                conv_instance.append_message(conv_instance.roles[0], question)
                conv_instance.append_message(conv_instance.roles[1], None)
                prompt = conv_instance.get_prompt()
                
                # Tokenize
                input_ids = tokenizer_point_token(prompt, tokenizer, POINT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                
                # Generate
                stop_str = conv_instance.sep if conv_instance.sep_style != SeparatorStyle.TWO else conv_instance.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                
                output_ids = model.generate(
                    input_ids,
                    points=point_cloud,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )
                
                # Decode output
                input_token_len = input_ids.shape[1]
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()
                
                # Store result
                result = {
                    'scene_id': anno['id'],
                    'point_file': anno['point'],
                    'question': question,
                    'ground_truth': ground_truth,
                    'model_output': outputs
                }
                results.append(result)
                
                # Print sample result every 20 samples
                if len(results) % 20 == 0:
                    print(f"\n--- Sample {len(results)} ---")
                    print(f"Scene: {anno['id']}")
                    print(f"Question: {question[:80]}...")
                    print(f"GT: {ground_truth[:80]}...")
                    print(f"Output: {outputs[:80]}...")
    
    return results


def calculate_metrics(results):
    """Calculate evaluation metrics for scaffold dataset"""
    print("\n" + "="*80)
    print("Calculating metrics...")
    print("="*80 + "\n")
    
    metrics = {
        'total_samples': len(results),
        'exact_match': 0,
        'bbox_questions': 0,
        'bbox_correct': 0,
        'missing_questions': 0,
        'missing_correct': 0,
        'why_questions': 0,
        'list_questions': 0,
        'list_correct': 0
    }
    
    exact_match = 0
    bbox_correct = 0
    bbox_total = 0
    missing_correct = 0
    missing_total = 0
    list_correct = 0
    list_total = 0
    why_total = 0
    
    for result in results:
        question = result['question'].lower()
        ground_truth = result['ground_truth']
        model_output = result['model_output']
        
        # Check question type
        is_bbox_question = 'referring' in question or 'return the box' in question
        is_missing_question = 'any missing' in question or 'check bay' in question
        is_why_question = 'why is' in question or 'why' in question
        is_list_question = 'list all missing' in question
        
        # Exact match
        if ground_truth.strip() == model_output.strip():
            exact_match += 1
            if is_bbox_question:
                bbox_correct += 1
            if is_missing_question:
                missing_correct += 1
            if is_list_question:
                list_correct += 1
        
        # Count by type
        if is_bbox_question:
            bbox_total += 1
        elif is_missing_question:
            missing_total += 1
        elif is_why_question:
            why_total += 1
        elif is_list_question:
            list_total += 1
        
        # Partial matching for JSON outputs
        try:
            if ground_truth.startswith('[') or ground_truth.startswith('{'):
                gt_json = json.loads(ground_truth)
                try:
                    output_json = json.loads(model_output)
                    
                    # For list questions
                    if is_list_question and isinstance(gt_json, list) and isinstance(output_json, list):
                        if len(gt_json) == len(output_json):
                            list_correct += 0.5
                except:
                    pass
        except:
            pass
    
    # Calculate metrics
    metrics['exact_match'] = exact_match
    metrics['exact_match_rate'] = (exact_match / len(results) * 100) if results else 0
    
    metrics['bbox_questions'] = bbox_total
    metrics['bbox_correct'] = bbox_correct
    metrics['bbox_accuracy'] = (bbox_correct / bbox_total * 100) if bbox_total > 0 else 0
    
    metrics['missing_questions'] = missing_total
    metrics['missing_correct'] = missing_correct
    metrics['missing_accuracy'] = (missing_correct / missing_total * 100) if missing_total > 0 else 0
    
    metrics['list_questions'] = list_total
    metrics['list_correct'] = list_correct
    metrics['list_accuracy'] = (list_correct / list_total * 100) if list_total > 0 else 0
    
    metrics['why_questions'] = why_total
    
    return metrics


def save_results(results, metrics, args):
    """Save evaluation results to JSON file"""
    output_path = os.path.join(args.output_dir, args.output_file)
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_data = {
        'model_path': args.model_path,
        'val_data_path': args.data_path,
        'val_anno_path': args.anno_path,
        'metrics': metrics,
        'results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


def print_metrics(metrics):
    """Print evaluation metrics"""
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Exact Match: {metrics['exact_match']} ({metrics['exact_match_rate']:.2f}%)")
    print("-" * 80)
    print(f"BBox Questions: {metrics['bbox_questions']}")
    print(f"  Correct: {metrics['bbox_correct']} ({metrics['bbox_accuracy']:.2f}%)")
    print("-" * 80)
    print(f"Missing Detection Questions: {metrics['missing_questions']}")
    print(f"  Correct: {metrics['missing_correct']} ({metrics['missing_accuracy']:.2f}%)")
    print("-" * 80)
    print(f"List Questions: {metrics['list_questions']}")
    print(f"  Correct: {metrics['list_correct']:.1f} ({metrics['list_accuracy']:.2f}%)")
    print("-" * 80)
    print(f"Why Questions: {metrics['why_questions']}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Scaffold dataset validation")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    # Dataset arguments
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to validation point cloud directory")
    parser.add_argument("--anno_path", type=str, required=True,
                        help="Path to validation annotation JSON file")
    parser.add_argument("--pointnum", type=int, default=10000)
    parser.add_argument("--use_color", action="store_true", default=False)
    
    # Data loader arguments
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="scaffold_val_results.json")
    
    args = parser.parse_args()
    
    # Initialize model
    print("Initializing model...")
    model, tokenizer, conv = init_model(args)
    
    # Load dataset
    print("Loading validation dataset...")
    dataset = ScaffoldValidationDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        pointnum=args.pointnum,
        use_color=args.use_color
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Run evaluation
    results = evaluate_scaffold(model, tokenizer, conv, dataloader, args)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save results
    save_results(results, metrics, args)
    
    print("\nValidation completed successfully!")


if __name__ == "__main__":
    main()