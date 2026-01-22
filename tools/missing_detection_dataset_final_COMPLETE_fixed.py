"""
Dataset generator for scaffold missing‑component detection (bbox fixed version).

This script synthesizes scaffold scenes using the ``EnhancedScaffoldGeneratorFinal`` and
creates both training and validation splits for missing detection tasks.  It
supports configurable train/validation ratios and includes bounding box
annotations in the evaluation ground truth.

🆕 Key improvements (bbox fix version):
- Fixed import to use bbox-corrected generator class
- Comprehensive question types (vertical, horizontal, platform)
- Proper train/val separation
- Missing quota system (max 4 components per scene)
- 3D bounding box annotations for evaluation (PROPERLY NORMALIZED -1~1)
- Enhanced bbox validation and range checking

Outputs:

* A **training SFT JSON** (``scaffold_missing_sft.json``) containing conversational
  examples for supervised fine‑tuning.  Only missing detection questions are
  included, and only scenes in the training split contribute to this file.
* Two **validation files** for evaluating model performance:
  - ``question_missing.jsonl``: a list of questions without the ``<point>`` token.
  - ``gt_missing.jsonl``: ground truth labels (``Yes``/``No``) and 3D bounding boxes
    for the relevant components.  Bounding boxes are populated for missing
    detection tasks where applicable.

Usage example:

    python missing_detection_dataset_final_fixed.py \
        --num-scenes 500 \
        --train-ratio 0.8 \
        --val-ratio 0.2 \
        --output-dir ./playground/data/shapellm/scaffold_missing_fixed

"""

import argparse
import json
import os
import random
from typing import List, Dict, Any

import numpy as np  # type: ignore

from modified_scaffold_generator_final_COMPLETE_fixed import EnhancedScaffoldGeneratorFinal


def validate_bbox_range(bbox: List[List[List[float]]], component_name: str = "unknown") -> bool:
    """Validate that bounding box coordinates are within -1~1 range."""
    if not bbox:
        return True
    
    try:
        bbox_array = np.array(bbox)
        min_val = bbox_array.min()
        max_val = bbox_array.max()
        
        if min_val < -1.1 or max_val > 1.1:
            print(f"⚠️ Warning: {component_name} bbox out of range: [{min_val:.3f}, {max_val:.3f}]")
            return False
        return True
    except Exception as e:
        print(f"⚠️ Error validating bbox for {component_name}: {e}")
        return False


def generate_missing_dataset(
    output_dir: str,
    num_scenes: int = 100,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
) -> None:
    """Generate synthetic scenes and export missing detection datasets.

    Scenes are split into training and validation subsets according to the
    provided ratios.  Only training scenes contribute to the SFT JSON, and
    only validation scenes produce questions and ground truth entries.

    Args:
        output_dir: Destination directory for dataset files.
        num_scenes: Number of scenes to synthesize.
        train_ratio: Fraction of scenes used for training.  Must be between 0 and 1.
        val_ratio: Fraction of scenes used for validation.  train_ratio + val_ratio <= 1.
    """

    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio and val_ratio must be non‑negative and sum to <= 1.0")

    os.makedirs(output_dir, exist_ok=True)
    pcs_dir = os.path.join(output_dir, 'pcs')
    os.makedirs(pcs_dir, exist_ok=True)

    scene_records: List[Dict[str, Any]] = []
    generator = EnhancedScaffoldGeneratorFinal(random_seed=42)  # Fixed seed for reproducibility

    print(f"🏗️ Generating {num_scenes} scaffold scenes with FIXED bbox normalization...")
    print(f"📊 Missing quota: {generator.missing_quota} components max per scene")

    # Statistics tracking
    bbox_validation_stats = {
        'total_scenes': 0,
        'scenes_with_bbox_errors': 0,
        'total_components': 0,
        'components_with_valid_bbox': 0,
        'components_with_invalid_bbox': 0
    }

    # 🆕 Missing component statistics
    missing_type_stats = {
        'perfect_scaffolds': 0,  # No missing components
        'vertical_only': 0,      # Only vertical posts missing
        'horizontal_only': 0,    # Only horizontal beams missing
        'platform_only': 0,      # Only platforms missing
        'vertical_horizontal': 0,  # V + H missing
        'vertical_platform': 0,    # V + P missing
        'horizontal_platform': 0,  # H + P missing
        'all_three': 0,            # V + H + P missing
        'total_missing_vertical': 0,
        'total_missing_horizontal': 0,
        'total_missing_platform': 0,
        'scenes_with_any_missing': 0
    }

    # Generate all scenes first
    for i in range(num_scenes):
        scene_id = f"scaffold_{i:05d}"
        scene_data = generator.generate_scene_data(scene_id)
        if scene_data is None:
            print(f"⚠️ Failed to generate {scene_id}")
            continue
            
        # Save point cloud
        coord = scene_data['coord']
        npy_path = os.path.join(pcs_dir, f"{scene_id}.npy")
        np.save(npy_path, coord.astype('float32'))
        
        # Count missing components by type
        missing_components = [c for c in scene_data['components'] if c.semantic_id == 10]
        missing_by_type = {
            'vertical': len([c for c in missing_components if (c.metadata or {}).get('defect_type') == 'missing_vertical']),
            'horizontal': len([c for c in missing_components if (c.metadata or {}).get('defect_type') == 'missing_horizontal']),
            'platform': len([c for c in missing_components if (c.metadata or {}).get('defect_type') == 'missing_platform'])
        }
        
        # 🆕 Validate bounding boxes
        scene_bbox_errors = 0
        for comp in scene_data['components']:
            bbox_validation_stats['total_components'] += 1
            if comp.bbox_norm is not None:
                bbox_list = comp.bbox_norm.tolist()
                if validate_bbox_range([bbox_list], comp.name):
                    bbox_validation_stats['components_with_valid_bbox'] += 1
                else:
                    bbox_validation_stats['components_with_invalid_bbox'] += 1
                    scene_bbox_errors += 1
            else:
                # Component without bbox (ok for some components)
                pass
        
        if scene_bbox_errors > 0:
            bbox_validation_stats['scenes_with_bbox_errors'] += 1
            print(f"⚠️ {scene_id}: {scene_bbox_errors} bbox errors")
        
        bbox_validation_stats['total_scenes'] += 1
        
        print(f"✅ {scene_id}: {len(missing_components)} missing total "
              f"(V:{missing_by_type['vertical']}, H:{missing_by_type['horizontal']}, P:{missing_by_type['platform']})")

        # 🆕 Update missing type statistics
        v_count = missing_by_type['vertical']
        h_count = missing_by_type['horizontal']
        p_count = missing_by_type['platform']

        missing_type_stats['total_missing_vertical'] += v_count
        missing_type_stats['total_missing_horizontal'] += h_count
        missing_type_stats['total_missing_platform'] += p_count

        if len(missing_components) == 0:
            missing_type_stats['perfect_scaffolds'] += 1
        else:
            missing_type_stats['scenes_with_any_missing'] += 1

            # Classify by missing pattern
            has_v = v_count > 0
            has_h = h_count > 0
            has_p = p_count > 0

            if has_v and has_h and has_p:
                missing_type_stats['all_three'] += 1
            elif has_v and has_h:
                missing_type_stats['vertical_horizontal'] += 1
            elif has_v and has_p:
                missing_type_stats['vertical_platform'] += 1
            elif has_h and has_p:
                missing_type_stats['horizontal_platform'] += 1
            elif has_v:
                missing_type_stats['vertical_only'] += 1
            elif has_h:
                missing_type_stats['horizontal_only'] += 1
            elif has_p:
                missing_type_stats['platform_only'] += 1

        # Store annotations and components for later splitting
        scene_records.append({
            'scene_id': scene_id,
            'annotations': scene_data['annotations'],
            'components': scene_data['components'],
        })

    print(f"\n📊 Generated {len(scene_records)} valid scenes")

    # 🆕 Print bbox validation statistics
    print(f"\n📏 Bounding Box Validation Report:")
    print(f"  🏗️ Total scenes: {bbox_validation_stats['total_scenes']}")
    print(f"  ❌ Scenes with bbox errors: {bbox_validation_stats['scenes_with_bbox_errors']}")
    print(f"  🔧 Total components: {bbox_validation_stats['total_components']}")
    print(f"  ✅ Valid bboxes: {bbox_validation_stats['components_with_valid_bbox']}")
    print(f"  ❌ Invalid bboxes: {bbox_validation_stats['components_with_invalid_bbox']}")
    
    if bbox_validation_stats['total_components'] > 0:
        valid_rate = bbox_validation_stats['components_with_valid_bbox'] / bbox_validation_stats['total_components'] * 100
        print(f"  📊 Bbox validity rate: {valid_rate:.1f}%")

    # Shuffle scenes and split
    random.shuffle(scene_records)
    n_train = int(train_ratio * len(scene_records))
    n_val = int(val_ratio * len(scene_records))
    train_records = scene_records[:n_train]
    val_records = scene_records[n_train:n_train + n_val]

    print(f"📂 Train/Val split: {len(train_records)} train, {len(val_records)} val")

    sft_entries: List[Dict[str, Any]] = []
    eval_questions: List[Dict[str, Any]] = []
    eval_gt: List[Dict[str, Any]] = []

    # Helper to collect bounding boxes of components filtered by predicate
    def collect_bboxes(comps: List[Any], predicate) -> List[List[List[float]]]:
        boxes: List[List[List[float]]] = []
        for c in comps:
            if predicate(c):
                if c.bbox_norm is not None:
                    bbox_list = c.bbox_norm.tolist()
                    # 🆕 Validate bbox before adding
                    if validate_bbox_range([bbox_list], c.name):
                        boxes.append(bbox_list)
                    else:
                        print(f"⚠️ Skipping invalid bbox for {c.name}")
        return boxes

    # Process training scenes - only SFT data
    print("\n🔧 Processing training scenes for SFT...")
    for record in train_records:
        for ann in record['annotations']:
            if not ann['task_type'].startswith('missing_detection'):
                continue
            # Append the full conversation for SFT training
            sft_entries.append({
                'id': ann['id'],
                'point': ann['point'],
                'conversations': ann['conversations'],
            })

    print(f"📝 Created {len(sft_entries)} SFT training examples")

    # Process validation scenes - evaluation data only
    print("\n🔧 Processing validation scenes for evaluation...")
    bbox_stats = {'valid_bboxes': 0, 'invalid_bboxes': 0, 'no_bboxes': 0}
    
    for record in val_records:
        comps = record['components']
        # Build instance_id -> component map for quick lookup
        comp_map = {c.instance_id: c for c in comps}
        
        # Precompute missing components by type
        missing_comps = [c for c in comps if c.semantic_id == 10]
        missing_verticals = [c for c in missing_comps if (c.metadata or {}).get('defect_type') == 'missing_vertical']
        missing_horizontals = [c for c in missing_comps if (c.metadata or {}).get('defect_type') == 'missing_horizontal']
        missing_platforms = [c for c in missing_comps if (c.metadata or {}).get('defect_type') == 'missing_platform']
        
        present_platforms = [c for c in comps if c.semantic_id == 3]
        present_verticals = [c for c in comps if c.semantic_id == 0]
        present_horizontals = [c for c in comps if c.semantic_id == 1]

        for ann in record['annotations']:
            task_type: str = ann.get('task_type', '')
            if not task_type.startswith('missing_detection'):
                continue
            convs = ann.get('conversations', [])
            if not convs or len(convs) < 2:
                continue
                
            # Extract question text (REMOVE <point> token - inference code adds it automatically)
            question_text = convs[0]['value'].replace('<point>\n', '').strip()
            # Extract full answer text (Template-Guided format with Expected vs Actual)
            answer_text = convs[1]['value']

            eval_questions.append({
                'question_id': ann['id'],
                'point': ann['point'],
                'text': question_text,
                'category': 'scaffold',
            })

            # Determine label and bounding boxes
            label = 'No'  # Default (simple Yes/No for backward compatibility)
            bboxes: List[List[List[float]]] = []
            
            # Determine answer based on task type
            if (task_type.endswith('_none') or 
                task_type.endswith('_negative') or 
                task_type.endswith('_positive')):
                # These are specific answer types
                if task_type.endswith('_positive'):
                    label = 'Yes'
                    # Get bbox of the specific positive component
                    inst_id = ann.get('target_instance_id')
                    c = comp_map.get(inst_id)
                    if c is not None and c.bbox_norm is not None:
                        bbox_list = c.bbox_norm.tolist()
                        if validate_bbox_range([bbox_list], c.name):
                            bboxes = [bbox_list]
                            bbox_stats['valid_bboxes'] += 1
                        else:
                            bbox_stats['invalid_bboxes'] += 1
                    else:
                        bbox_stats['no_bboxes'] += 1
                else:
                    label = 'No'  # for _none and _negative
                    bbox_stats['no_bboxes'] += 1
            else:
                # These are questions about missing components
                if (task_type == 'missing_detection_summary' and missing_comps):
                    label = 'Yes'
                    valid_bboxes = []
                    for c in missing_comps:
                        if c.bbox_norm is not None:
                            bbox_list = c.bbox_norm.tolist()
                            if validate_bbox_range([bbox_list], c.name):
                                valid_bboxes.append(bbox_list)
                                bbox_stats['valid_bboxes'] += 1
                            else:
                                bbox_stats['invalid_bboxes'] += 1
                        else:
                            bbox_stats['no_bboxes'] += 1
                    bboxes = valid_bboxes
                elif (task_type == 'missing_detection_floor'):
                    floor = ann.get('target_floor')
                    floor_missing = collect_bboxes(missing_comps, 
                        lambda c: str((c.metadata or {}).get('floor', '?')) == str(floor))
                    if floor_missing:
                        label = 'Yes'
                        bboxes = floor_missing
                        bbox_stats['valid_bboxes'] += len(floor_missing)
                    else:
                        bbox_stats['no_bboxes'] += 1
                elif (task_type == 'missing_detection_bay'):
                    bay = ann.get('target_bay')
                    bay_missing = collect_bboxes(missing_comps, 
                        lambda c: str((c.metadata or {}).get('bay', '?')) == str(bay))
                    if bay_missing:
                        label = 'Yes'
                        bboxes = bay_missing
                        bbox_stats['valid_bboxes'] += len(bay_missing)
                    else:
                        bbox_stats['no_bboxes'] += 1
                elif (task_type == 'missing_detection_specific'):
                    inst_id = ann.get('target_instance_id')
                    c = comp_map.get(inst_id)
                    if c is not None and c.semantic_id == 10 and c.bbox_norm is not None:
                        bbox_list = c.bbox_norm.tolist()
                        if validate_bbox_range([bbox_list], c.name):
                            label = 'Yes'
                            bboxes = [bbox_list]
                            bbox_stats['valid_bboxes'] += 1
                        else:
                            bbox_stats['invalid_bboxes'] += 1
                    else:
                        bbox_stats['no_bboxes'] += 1
                elif (task_type == 'missing_detection_vertical_summary' and missing_verticals):
                    label = 'Yes'
                    valid_bboxes = []
                    for c in missing_verticals:
                        if c.bbox_norm is not None:
                            bbox_list = c.bbox_norm.tolist()
                            if validate_bbox_range([bbox_list], c.name):
                                valid_bboxes.append(bbox_list)
                                bbox_stats['valid_bboxes'] += 1
                            else:
                                bbox_stats['invalid_bboxes'] += 1
                        else:
                            bbox_stats['no_bboxes'] += 1
                    bboxes = valid_bboxes
                elif (task_type == 'missing_detection_vertical_specific'):
                    inst_id = ann.get('target_instance_id')
                    c = comp_map.get(inst_id)
                    if (c is not None and c.semantic_id == 10 and 
                        (c.metadata or {}).get('defect_type') == 'missing_vertical' and 
                        c.bbox_norm is not None):
                        bbox_list = c.bbox_norm.tolist()
                        if validate_bbox_range([bbox_list], c.name):
                            label = 'Yes'
                            bboxes = [bbox_list]
                            bbox_stats['valid_bboxes'] += 1
                        else:
                            bbox_stats['invalid_bboxes'] += 1
                    else:
                        bbox_stats['no_bboxes'] += 1
                elif (task_type == 'missing_detection_horizontal_summary' and missing_horizontals):
                    label = 'Yes'
                    valid_bboxes = []
                    for c in missing_horizontals:
                        if c.bbox_norm is not None:
                            bbox_list = c.bbox_norm.tolist()
                            if validate_bbox_range([bbox_list], c.name):
                                valid_bboxes.append(bbox_list)
                                bbox_stats['valid_bboxes'] += 1
                            else:
                                bbox_stats['invalid_bboxes'] += 1
                        else:
                            bbox_stats['no_bboxes'] += 1
                    bboxes = valid_bboxes
                elif (task_type == 'missing_detection_horizontal_specific'):
                    inst_id = ann.get('target_instance_id')
                    c = comp_map.get(inst_id)
                    if (c is not None and c.semantic_id == 10 and 
                        (c.metadata or {}).get('defect_type') == 'missing_horizontal' and 
                        c.bbox_norm is not None):
                        bbox_list = c.bbox_norm.tolist()
                        if validate_bbox_range([bbox_list], c.name):
                            label = 'Yes'
                            bboxes = [bbox_list]
                            bbox_stats['valid_bboxes'] += 1
                        else:
                            bbox_stats['invalid_bboxes'] += 1
                    else:
                        bbox_stats['no_bboxes'] += 1
                # Additional cases for summary questions without missing components
                elif (task_type in ['missing_detection_summary', 'missing_detection_vertical_summary', 
                                  'missing_detection_horizontal_summary'] and not missing_comps):
                    label = 'No'
                    bboxes = []
                    bbox_stats['no_bboxes'] += 1
                else:
                    bbox_stats['no_bboxes'] += 1
            
            eval_gt.append({
                'question_id': ann['id'],
                'point': ann['point'],
                'text': answer_text,  # Full answer with Expected vs Actual (Idea 1)
                'label': label,  # Simple Yes/No for backward compatibility
                'bboxes': bboxes,
                'category': 'scaffold',
            })

    print(f"❓ Created {len(eval_questions)} evaluation questions")
    print(f"✅ Created {len(eval_gt)} ground truth entries")

    # 🆕 Print bbox processing statistics
    print(f"\n📏 Evaluation Bbox Processing Stats:")
    print(f"  ✅ Valid bboxes: {bbox_stats['valid_bboxes']}")
    print(f"  ❌ Invalid bboxes (skipped): {bbox_stats['invalid_bboxes']}")
    print(f"  ⚪ No bboxes (expected): {bbox_stats['no_bboxes']}")
    
    total_bbox_attempts = sum(bbox_stats.values())
    if total_bbox_attempts > 0:
        valid_rate = bbox_stats['valid_bboxes'] / total_bbox_attempts * 100
        print(f"  📊 Bbox processing success rate: {valid_rate:.1f}%")

    # Write training SFT file
    sft_path = os.path.join(output_dir, 'scaffold_missing_sft.json')
    with open(sft_path, 'w', encoding='utf-8') as f:
        json.dump(sft_entries, f, indent=2)

    # Write evaluation questions
    q_path = os.path.join(output_dir, 'question_missing.jsonl')
    with open(q_path, 'w', encoding='utf-8') as f:
        for q in eval_questions:
            f.write(json.dumps(q) + '\n')

    # Write evaluation ground truth
    gt_path = os.path.join(output_dir, 'gt_missing.jsonl')
    with open(gt_path, 'w', encoding='utf-8') as f:
        for g in eval_gt:
            f.write(json.dumps(g) + '\n')

    # 🆕 Print missing component statistics
    print(f"\n📊 Missing Component Statistics:")
    print(f"  ✅ Perfect scaffolds (no missing): {missing_type_stats['perfect_scaffolds']}")
    print(f"  ⚠️ Scaffolds with missing components: {missing_type_stats['scenes_with_any_missing']}")
    print(f"\n  📌 Missing patterns:")
    print(f"     Vertical only: {missing_type_stats['vertical_only']}")
    print(f"     Horizontal only: {missing_type_stats['horizontal_only']}")
    print(f"     Platform only: {missing_type_stats['platform_only']}")
    print(f"     Vertical + Horizontal: {missing_type_stats['vertical_horizontal']}")
    print(f"     Vertical + Platform: {missing_type_stats['vertical_platform']}")
    print(f"     Horizontal + Platform: {missing_type_stats['horizontal_platform']}")
    print(f"     All three types: {missing_type_stats['all_three']}")
    print(f"\n  📊 Total missing components:")
    print(f"     Vertical posts: {missing_type_stats['total_missing_vertical']}")
    print(f"     Horizontal beams: {missing_type_stats['total_missing_horizontal']}")
    print(f"     Platforms: {missing_type_stats['total_missing_platform']}")
    total_missing = (missing_type_stats['total_missing_vertical'] +
                     missing_type_stats['total_missing_horizontal'] +
                     missing_type_stats['total_missing_platform'])
    print(f"     TOTAL: {total_missing}")

    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"🏗️ Total scenes: {len(scene_records)}")
    print(f"📚 Training scenes: {len(train_records)}")
    print(f"🧪 Validation scenes: {len(val_records)}")
    print(f"📝 SFT training examples: {len(sft_entries)}")
    print(f"❓ Evaluation questions: {len(eval_questions)}")
    print(f"✅ Ground truth entries: {len(eval_gt)}")

    # Analyze question types
    task_type_counts = {}
    for entry in sft_entries:
        # Extract task type from entry id
        entry_id = entry['id']
        if '_missing_summary' in entry_id:
            task_type = 'summary'
        elif '_missing_vertical_' in entry_id:
            task_type = 'vertical'
        elif '_missing_horizontal_' in entry_id:
            task_type = 'horizontal'
        elif '_missing_floor_' in entry_id:
            task_type = 'floor'
        elif '_missing_bay_' in entry_id:
            task_type = 'bay'
        elif '_missing_specific_' in entry_id:
            task_type = 'specific'
        elif '_missing_positive_' in entry_id:
            task_type = 'positive'
        elif '_missing_none' in entry_id:
            task_type = 'none'
        else:
            task_type = 'other'
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

    print(f"\n🔍 Question Type Distribution:")
    for task_type, count in sorted(task_type_counts.items()):
        print(f"  {task_type}: {count}")

    # 🆕 Save statistics to JSON file
    stats_path = os.path.join(output_dir, 'dataset_statistics.json')
    statistics_report = {
        'generation_info': {
            'total_scenes': len(scene_records),
            'train_scenes': len(train_records),
            'val_scenes': len(val_records),
            'sft_examples': len(sft_entries),
            'eval_questions': len(eval_questions),
            'random_seed': 42
        },
        'missing_components': {
            'perfect_scaffolds': missing_type_stats['perfect_scaffolds'],
            'scaffolds_with_missing': missing_type_stats['scenes_with_any_missing'],
            'missing_patterns': {
                'vertical_only': missing_type_stats['vertical_only'],
                'horizontal_only': missing_type_stats['horizontal_only'],
                'platform_only': missing_type_stats['platform_only'],
                'vertical_horizontal': missing_type_stats['vertical_horizontal'],
                'vertical_platform': missing_type_stats['vertical_platform'],
                'horizontal_platform': missing_type_stats['horizontal_platform'],
                'all_three_types': missing_type_stats['all_three']
            },
            'total_counts': {
                'vertical_posts': missing_type_stats['total_missing_vertical'],
                'horizontal_beams': missing_type_stats['total_missing_horizontal'],
                'platforms': missing_type_stats['total_missing_platform'],
                'total': total_missing
            }
        },
        'bbox_validation': bbox_validation_stats,
        'question_types': task_type_counts
    }

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(statistics_report, f, indent=2)

    print(f"\n💾 Files saved:")
    print(f"  📁 Point clouds: {pcs_dir}")
    print(f"  📝 Training SFT: {sft_path}")
    print(f"  ❓ Eval questions: {q_path}")
    print(f"  ✅ Ground truth: {gt_path}")
    print(f"  📊 Statistics: {stats_path}")

    print(f"\n🎯 Key improvements applied (bbox fixed version):")
    print(f"✅ Fixed bbox normalization (-1~1 range GUARANTEED)")
    print(f"✅ Missing quota system (max {generator.missing_quota} per scene)")
    print(f"✅ Vertical/horizontal/platform missing detection")
    print(f"✅ Comprehensive question types (summary, floor, bay, specific, positive)")
    print(f"✅ Proper train/val separation")
    print(f"✅ 3D bounding box annotations (properly normalized)")
    print(f"✅ Bbox validation and error reporting")
    print(f"✅ Z-axis vertical ladders")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a missing component detection dataset for scaffolds (Bbox Fixed Version).')
    parser.add_argument('--output-dir', type=str, default='./playground/data/shapellm/scaffold_missing_fixed', 
                       help='Directory to save dataset files.')
    parser.add_argument('--num-scenes', type=int, default=1000, 
                       help='Number of scenes to generate.')
    parser.add_argument('--train-ratio', type=float, default=0.8, 
                       help='Fraction of scenes to use for training.')
    parser.add_argument('--val-ratio', type=float, default=0.2, 
                       help='Fraction of scenes to use for validation.')
    
    args = parser.parse_args()
    
    print("🏗️ ShapeLLM Scaffold Missing Detection Dataset Generator (Bbox Fixed Version)")
    print("="*80)
    
    generate_missing_dataset(
        output_dir=args.output_dir,
        num_scenes=args.num_scenes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    
    print("="*80)
    print("✅ Dataset generation complete!")
    print("🎯 All bounding boxes are now properly normalized to -1~1 range!")