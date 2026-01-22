"""
Dataset Builder Module

This module orchestrates the complete data generation pipeline,
producing properly split datasets for Stage 1 and Stage 2 training.
"""

import os
import json
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

from .config import ScaffoldConfig
from .scaffold_generator import ScaffoldGenerator, ScaffoldComponent
from .question_generator import QuestionGenerator, QAPair, CaptionGenerator


@dataclass
class DatasetStatistics:
    """Statistics for generated dataset."""

    total_scenes: int = 0
    train_scenes: int = 0
    val_scenes: int = 0
    test_scenes: int = 0

    # Missing component stats
    scenes_with_missing: int = 0
    total_missing_vertical: int = 0
    total_missing_horizontal: int = 0
    total_missing_platform: int = 0

    # QA stats
    stage1_train_examples: int = 0
    stage1_val_examples: int = 0
    stage2_train_examples: int = 0
    stage2_val_examples: int = 0
    test_questions: int = 0

    # Label distribution
    yes_labels: int = 0
    no_labels: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scenes': {
                'total': self.total_scenes,
                'train': self.train_scenes,
                'val': self.val_scenes,
                'test': self.test_scenes,
                'with_missing': self.scenes_with_missing
            },
            'missing_components': {
                'vertical': self.total_missing_vertical,
                'horizontal': self.total_missing_horizontal,
                'platform': self.total_missing_platform,
                'total': self.total_missing_vertical + self.total_missing_horizontal + self.total_missing_platform
            },
            'examples': {
                'stage1_train': self.stage1_train_examples,
                'stage1_val': self.stage1_val_examples,
                'stage2_train': self.stage2_train_examples,
                'stage2_val': self.stage2_val_examples,
                'test': self.test_questions
            },
            'label_distribution': {
                'yes': self.yes_labels,
                'no': self.no_labels,
                'balance_ratio': self.yes_labels / max(self.no_labels, 1)
            }
        }


class DatasetBuilder:
    """
    Orchestrates the complete dataset generation pipeline.

    Produces:
    - Stage 1 training data (captions for feature alignment)
    - Stage 2 training data (instruction following)
    - Validation data
    - Test data with ground truth
    """

    def __init__(self, config: ScaffoldConfig):
        """Initialize with configuration."""
        self.config = config
        self.scaffold_gen = ScaffoldGenerator(config)
        self.question_gen = QuestionGenerator(config)
        self.caption_gen = CaptionGenerator(config)
        self.stats = DatasetStatistics()

    def _ensure_dirs(self) -> str:
        """Create output directories and return base path."""
        base_dir = self.config.dataset.output_dir
        pcs_dir = os.path.join(base_dir, self.config.dataset.pcs_dir)

        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(pcs_dir, exist_ok=True)

        return base_dir

    def _split_scenes(
        self,
        scene_ids: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split scene IDs into train/val/test."""
        random.shuffle(scene_ids)

        n = len(scene_ids)
        n_train = int(n * self.config.dataset.train_ratio)
        n_val = int(n * self.config.dataset.val_ratio)

        train_ids = scene_ids[:n_train]
        val_ids = scene_ids[n_train:n_train + n_val]
        test_ids = scene_ids[n_train + n_val:]

        return train_ids, val_ids, test_ids

    def _save_point_cloud(
        self,
        scene_data: Dict[str, Any],
        pcs_dir: str
    ) -> None:
        """Save point cloud to disk."""
        scene_id = scene_data['scene_id']
        coord = scene_data['coord']

        npy_path = os.path.join(pcs_dir, f"{scene_id}.npy")
        np.save(npy_path, coord.astype('float32'))

    def _update_missing_stats(
        self,
        components: List[ScaffoldComponent]
    ) -> None:
        """Update missing component statistics."""
        missing_comps = [c for c in components if c.semantic_id == 10]

        if missing_comps:
            self.stats.scenes_with_missing += 1

            for comp in missing_comps:
                defect_type = (comp.metadata or {}).get('defect_type', '')
                if defect_type == 'missing_vertical':
                    self.stats.total_missing_vertical += 1
                elif defect_type == 'missing_horizontal':
                    self.stats.total_missing_horizontal += 1
                elif defect_type == 'missing_platform':
                    self.stats.total_missing_platform += 1

    def generate_dataset(self) -> DatasetStatistics:
        """
        Generate the complete dataset.

        Returns:
            DatasetStatistics with generation results
        """
        print("=" * 80)
        print("ShapeLLM Scaffold Dataset Generator v2.0")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  - Scenes: {self.config.num_scenes}")
        print(f"  - Split: {self.config.dataset.train_ratio:.0%} train, "
              f"{self.config.dataset.val_ratio:.0%} val, "
              f"{self.config.dataset.test_ratio:.0%} test")
        print(f"  - Output: {self.config.dataset.output_dir}")
        print("=" * 80)

        base_dir = self._ensure_dirs()
        pcs_dir = os.path.join(base_dir, self.config.dataset.pcs_dir)

        # Phase 1: Generate all scenes
        print("\n[Phase 1/3] Generating scaffold scenes...")
        all_scenes = []
        scene_ids = []

        for i in tqdm(range(self.config.num_scenes), desc="Generating"):
            scene_id = f"scaffold_{i:05d}"
            scene_data = self.scaffold_gen.generate_scene(scene_id)

            if scene_data is not None:
                all_scenes.append(scene_data)
                scene_ids.append(scene_id)
                self._save_point_cloud(scene_data, pcs_dir)
                self._update_missing_stats(scene_data['components'])

        self.stats.total_scenes = len(all_scenes)
        print(f"  Generated {len(all_scenes)} valid scenes")

        # Phase 2: Split scenes
        print("\n[Phase 2/3] Splitting dataset...")
        train_ids, val_ids, test_ids = self._split_scenes(scene_ids)

        self.stats.train_scenes = len(train_ids)
        self.stats.val_scenes = len(val_ids)
        self.stats.test_scenes = len(test_ids)

        print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

        # Create ID to scene mapping
        scene_map = {s['scene_id']: s for s in all_scenes}

        # Phase 3: Generate QA data
        print("\n[Phase 3/3] Generating QA pairs...")

        # Stage 1 data (captions)
        stage1_train = []
        stage1_val = []

        for scene_id in tqdm(train_ids, desc="Stage1 Train"):
            scene = scene_map[scene_id]
            captions = self.caption_gen.generate_captions(
                scene_id, scene['components'], scene['config']
            )
            stage1_train.extend(captions)

        for scene_id in tqdm(val_ids, desc="Stage1 Val"):
            scene = scene_map[scene_id]
            captions = self.caption_gen.generate_captions(
                scene_id, scene['components'], scene['config']
            )
            stage1_val.extend(captions)

        self.stats.stage1_train_examples = len(stage1_train)
        self.stats.stage1_val_examples = len(stage1_val)

        # Stage 2 data (instruction following)
        stage2_train = []
        stage2_val = []

        for scene_id in tqdm(train_ids, desc="Stage2 Train"):
            scene = scene_map[scene_id]
            qa_pairs = self.question_gen.generate_all_qa_pairs(
                scene_id, scene['components'], scene['config']
            )
            for qa in qa_pairs:
                stage2_train.append(qa.to_sft_format())
                if qa.label == 'Yes':
                    self.stats.yes_labels += 1
                else:
                    self.stats.no_labels += 1

        for scene_id in tqdm(val_ids, desc="Stage2 Val"):
            scene = scene_map[scene_id]
            qa_pairs = self.question_gen.generate_all_qa_pairs(
                scene_id, scene['components'], scene['config']
            )
            for qa in qa_pairs:
                stage2_val.append(qa.to_sft_format())

        self.stats.stage2_train_examples = len(stage2_train)
        self.stats.stage2_val_examples = len(stage2_val)

        # Test data (questions + ground truth)
        test_questions = []
        test_gt = []

        for scene_id in tqdm(test_ids, desc="Test Set"):
            scene = scene_map[scene_id]
            qa_pairs = self.question_gen.generate_all_qa_pairs(
                scene_id, scene['components'], scene['config']
            )
            for qa in qa_pairs:
                test_questions.append(qa.to_eval_question())
                test_gt.append(qa.to_eval_gt())

        self.stats.test_questions = len(test_questions)

        # Save all files
        print("\n[Saving] Writing output files...")

        # Stage 1
        with open(os.path.join(base_dir, self.config.dataset.stage1_train_file), 'w') as f:
            json.dump(stage1_train, f, indent=2)
        with open(os.path.join(base_dir, self.config.dataset.stage1_val_file), 'w') as f:
            json.dump(stage1_val, f, indent=2)

        # Stage 2
        with open(os.path.join(base_dir, self.config.dataset.stage2_train_file), 'w') as f:
            json.dump(stage2_train, f, indent=2)
        with open(os.path.join(base_dir, self.config.dataset.stage2_val_file), 'w') as f:
            json.dump(stage2_val, f, indent=2)

        # Test
        with open(os.path.join(base_dir, self.config.dataset.test_questions_file), 'w') as f:
            for q in test_questions:
                f.write(json.dumps(q) + '\n')
        with open(os.path.join(base_dir, self.config.dataset.test_gt_file), 'w') as f:
            for g in test_gt:
                f.write(json.dumps(g) + '\n')

        # Statistics
        stats_path = os.path.join(base_dir, 'dataset_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)

        # Config backup
        self.config.to_yaml(os.path.join(base_dir, 'config.yaml'))

        print("\n" + "=" * 80)
        print("Dataset Generation Complete!")
        print("=" * 80)
        print(f"Output directory: {base_dir}")
        print(f"\nFiles generated:")
        print(f"  - {self.config.dataset.stage1_train_file}: {len(stage1_train)} examples")
        print(f"  - {self.config.dataset.stage1_val_file}: {len(stage1_val)} examples")
        print(f"  - {self.config.dataset.stage2_train_file}: {len(stage2_train)} examples")
        print(f"  - {self.config.dataset.stage2_val_file}: {len(stage2_val)} examples")
        print(f"  - {self.config.dataset.test_questions_file}: {len(test_questions)} questions")
        print(f"  - {self.config.dataset.test_gt_file}: {len(test_gt)} ground truth entries")
        print(f"\nLabel distribution:")
        print(f"  - Yes: {self.stats.yes_labels}")
        print(f"  - No: {self.stats.no_labels}")
        print(f"  - Balance ratio: {self.stats.yes_labels / max(self.stats.no_labels, 1):.2f}")
        print("=" * 80)

        return self.stats


class AblationDataGenerator:
    """
    Generator for ablation study datasets.

    These datasets help verify that the model actually uses
    point cloud information rather than text shortcuts.
    """

    def __init__(self, config: ScaffoldConfig, base_dataset_dir: str):
        """Initialize with configuration and base dataset path."""
        self.config = config
        self.base_dir = base_dataset_dir
        self.ablation_config = config.ablation

    def generate_noise_ablation(self) -> str:
        """
        Generate random noise point clouds with same questions.

        If model performs similarly on noise vs real point clouds,
        it indicates the model is not using visual information.
        """
        ablation_dir = os.path.join(self.base_dir, 'ablation_noise')
        os.makedirs(ablation_dir, exist_ok=True)
        pcs_dir = os.path.join(ablation_dir, 'pcs')
        os.makedirs(pcs_dir, exist_ok=True)

        # Load test questions
        test_q_path = os.path.join(
            self.base_dir,
            self.config.dataset.test_questions_file
        )

        questions = []
        with open(test_q_path, 'r') as f:
            for line in f:
                questions.append(json.loads(line))

        # Sample and generate noise versions
        sampled = random.sample(
            questions,
            min(self.ablation_config.samples_per_type, len(questions))
        )

        ablation_questions = []
        for q in sampled:
            # Generate random noise point cloud
            noise_pc = np.random.randn(10000, 6).astype('float32')
            noise_pc[:, :3] = np.clip(noise_pc[:, :3], -1, 1)  # XYZ in [-1, 1]
            noise_pc[:, 3:] = np.clip(noise_pc[:, 3:] * 0.5 + 0.5, 0, 1)  # RGB in [0, 1]

            # Save with new name
            new_point_file = q['point'].replace('.npy', '_noise.npy')
            np.save(os.path.join(pcs_dir, new_point_file), noise_pc)

            # Update question
            ablation_q = q.copy()
            ablation_q['point'] = new_point_file
            ablation_q['question_id'] = q['question_id'] + '_noise'
            ablation_questions.append(ablation_q)

        # Save ablation questions
        q_path = os.path.join(ablation_dir, 'questions.jsonl')
        with open(q_path, 'w') as f:
            for q in ablation_questions:
                f.write(json.dumps(q) + '\n')

        print(f"Generated noise ablation: {len(ablation_questions)} samples")
        return ablation_dir

    def generate_shuffled_ablation(self) -> str:
        """
        Generate shuffled point clouds (same points, random order).

        Point order shouldn't matter for a properly trained model,
        but this tests robustness.
        """
        ablation_dir = os.path.join(self.base_dir, 'ablation_shuffled')
        os.makedirs(ablation_dir, exist_ok=True)
        pcs_dir = os.path.join(ablation_dir, 'pcs')
        os.makedirs(pcs_dir, exist_ok=True)

        # Load original point clouds
        orig_pcs_dir = os.path.join(self.base_dir, self.config.dataset.pcs_dir)

        # Load test questions
        test_q_path = os.path.join(
            self.base_dir,
            self.config.dataset.test_questions_file
        )

        questions = []
        with open(test_q_path, 'r') as f:
            for line in f:
                questions.append(json.loads(line))

        # Get unique point files
        unique_files = list(set(q['point'] for q in questions))
        sampled_files = random.sample(
            unique_files,
            min(self.ablation_config.samples_per_type, len(unique_files))
        )

        # Shuffle and save
        for pf in sampled_files:
            orig_path = os.path.join(orig_pcs_dir, pf)
            if os.path.exists(orig_path):
                pc = np.load(orig_path)
                np.random.shuffle(pc)  # Shuffle in place

                new_name = pf.replace('.npy', '_shuffled.npy')
                np.save(os.path.join(pcs_dir, new_name), pc)

        # Update questions
        ablation_questions = []
        for q in questions:
            if q['point'] in sampled_files:
                ablation_q = q.copy()
                ablation_q['point'] = q['point'].replace('.npy', '_shuffled.npy')
                ablation_q['question_id'] = q['question_id'] + '_shuffled'
                ablation_questions.append(ablation_q)

        # Save
        q_path = os.path.join(ablation_dir, 'questions.jsonl')
        with open(q_path, 'w') as f:
            for q in ablation_questions:
                f.write(json.dumps(q) + '\n')

        print(f"Generated shuffled ablation: {len(ablation_questions)} samples")
        return ablation_dir

    def generate_cross_scaffold_ablation(self) -> str:
        """
        Generate mismatched scaffold-question pairs.

        Use questions from one scaffold with point cloud from another.
        If model relies on text patterns, it will still answer "correctly"
        based on question text.
        """
        ablation_dir = os.path.join(self.base_dir, 'ablation_cross')
        os.makedirs(ablation_dir, exist_ok=True)

        # Load test questions
        test_q_path = os.path.join(
            self.base_dir,
            self.config.dataset.test_questions_file
        )

        questions = []
        with open(test_q_path, 'r') as f:
            for line in f:
                questions.append(json.loads(line))

        # Group by point file
        by_point = {}
        for q in questions:
            pf = q['point']
            if pf not in by_point:
                by_point[pf] = []
            by_point[pf].append(q)

        point_files = list(by_point.keys())

        if len(point_files) < 2:
            print("Not enough point files for cross-scaffold ablation")
            return ablation_dir

        # Create mismatched pairs
        ablation_questions = []
        n_samples = min(self.ablation_config.samples_per_type, len(questions))

        for _ in range(n_samples):
            # Pick random question
            q = random.choice(questions)

            # Pick different point file
            other_pf = random.choice([pf for pf in point_files if pf != q['point']])

            ablation_q = q.copy()
            ablation_q['point'] = other_pf  # Wrong point cloud!
            ablation_q['question_id'] = q['question_id'] + '_cross'
            ablation_q['_original_point'] = q['point']  # Keep for analysis
            ablation_questions.append(ablation_q)

        # Save (uses original point clouds, just mismatched)
        q_path = os.path.join(ablation_dir, 'questions.jsonl')
        with open(q_path, 'w') as f:
            for q in ablation_questions:
                f.write(json.dumps(q) + '\n')

        print(f"Generated cross-scaffold ablation: {len(ablation_questions)} samples")
        return ablation_dir

    def generate_all_ablations(self) -> Dict[str, str]:
        """Generate all ablation datasets."""
        print("\n[Ablation] Generating ablation study datasets...")

        results = {}

        if 'random_noise' in self.ablation_config.ablation_types:
            results['noise'] = self.generate_noise_ablation()

        if 'shuffled' in self.ablation_config.ablation_types:
            results['shuffled'] = self.generate_shuffled_ablation()

        if 'different_scaffold' in self.ablation_config.ablation_types:
            results['cross'] = self.generate_cross_scaffold_ablation()

        print(f"Generated {len(results)} ablation datasets")
        return results
