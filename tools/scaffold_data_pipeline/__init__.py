"""
Scaffold Data Pipeline for ShapeLLM Training

This module provides a modular, academically rigorous data generation pipeline
for scaffold missing detection tasks.

Key Features:
- Data Leakage Prevention: Uniform question formulation
- Two-Stage Training Support: Stage 1 (Caption) + Stage 2 (Instruction)
- Ablation Study Tools: Point cloud dependency verification
- Academic Rigor: Proper train/val/test splits, reproducibility

Usage:
    python -m tools.scaffold_data_pipeline.main --config config.yaml

Author: Research Team
Date: 2026-01-22
"""

from .config import ScaffoldConfig
from .scaffold_generator import ScaffoldGenerator
from .question_generator import QuestionGenerator
from .dataset_builder import DatasetBuilder

__all__ = [
    'ScaffoldConfig',
    'ScaffoldGenerator',
    'QuestionGenerator',
    'DatasetBuilder'
]
