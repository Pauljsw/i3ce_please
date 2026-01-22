"""
Question Generator Module - CRITICAL for preventing Data Leakage

This module generates questions and answers for scaffold missing detection.
All questions use UNIFORM templates to prevent the model from learning
shortcuts based on question text patterns.

Key Principle:
- Same question format for both Yes and No cases
- Model must rely on point cloud analysis, not text patterns
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .config import ScaffoldConfig, QuestionConfig, AnswerConfig
from .scaffold_generator import ScaffoldComponent


@dataclass
class QAPair:
    """Question-Answer pair with metadata."""

    question_id: str
    point_file: str
    question: str
    answer: str
    label: str  # 'Yes' or 'No'
    bboxes: List[List[List[float]]]  # List of 8-corner bboxes
    task_type: str
    metadata: Dict[str, Any]

    def to_sft_format(self) -> Dict[str, Any]:
        """Convert to SFT training format."""
        return {
            'id': self.question_id,
            'point': self.point_file,
            'conversations': [
                {'from': 'human', 'value': f'<point>\n{self.question}'},
                {'from': 'gpt', 'value': self.answer}
            ]
        }

    def to_eval_question(self) -> Dict[str, Any]:
        """Convert to evaluation question format."""
        return {
            'question_id': self.question_id,
            'point': self.point_file,
            'text': self.question,
            'category': 'scaffold'
        }

    def to_eval_gt(self) -> Dict[str, Any]:
        """Convert to evaluation ground truth format."""
        return {
            'question_id': self.question_id,
            'point': self.point_file,
            'text': self.answer,
            'label': self.label,
            'bboxes': self.bboxes,
            'category': 'scaffold',
            'task_type': self.task_type
        }


class QuestionGenerator:
    """
    Generator for scaffold-related questions and answers.

    CRITICAL DESIGN PRINCIPLE:
    All question templates are UNIFORM regardless of whether the answer is Yes or No.
    This prevents data leakage through question text patterns.
    """

    def __init__(self, config: ScaffoldConfig):
        """Initialize with configuration."""
        self.config = config
        self.q_config = config.question
        self.a_config = config.answer

    def _format_bbox(self, bbox_norm: Optional[Any]) -> str:
        """Format normalized bounding box for text output."""
        if bbox_norm is None:
            return "[[N/A]]"

        try:
            import numpy as np
            if isinstance(bbox_norm, np.ndarray):
                bbox_list = bbox_norm.tolist()
            else:
                bbox_list = bbox_norm

            # Format as 8-corner bbox
            formatted = "["
            for i, corner in enumerate(bbox_list):
                formatted += f"[{corner[0]:.3f}, {corner[1]:.3f}, {corner[2]:.3f}]"
                if i < len(bbox_list) - 1:
                    formatted += ", "
            formatted += "]"
            return formatted
        except Exception:
            return "[[Error]]"

    def _get_scaffold_spec(self, config: Dict[str, Any]) -> str:
        """Generate human-readable scaffold specification."""
        return f"{config['num_bays']}-bay, {config['num_floors']}-floor scaffold"

    def _generate_missing_details(
        self,
        missing_components: List[ScaffoldComponent]
    ) -> str:
        """Generate detailed description of missing components."""
        details = []
        for comp in missing_components[:5]:  # Limit to 5 for readability
            metadata = comp.metadata or {}
            defect_type = metadata.get('defect_type', 'unknown')

            if defect_type == 'missing_vertical':
                col = metadata.get('column', '?')
                row = metadata.get('row', '?')
                details.append(
                    f"- Vertical post at column {col}, row {row}: "
                    f"{self._format_bbox(comp.bbox_norm)}"
                )
            elif defect_type == 'missing_horizontal':
                floor = metadata.get('floor', '?')
                orientation = metadata.get('orientation', '?')
                if orientation == 'X':
                    bay = metadata.get('bay', '?')
                    details.append(
                        f"- Horizontal beam (X) at floor {floor}, bay {bay}: "
                        f"{self._format_bbox(comp.bbox_norm)}"
                    )
                else:
                    col = metadata.get('column', '?')
                    details.append(
                        f"- Horizontal beam (Y) at floor {floor}, column {col}: "
                        f"{self._format_bbox(comp.bbox_norm)}"
                    )
            elif defect_type == 'missing_platform':
                floor = metadata.get('floor', '?')
                bay = metadata.get('bay', '?')
                details.append(
                    f"- Platform at floor {floor}, bay {bay}: "
                    f"{self._format_bbox(comp.bbox_norm)}"
                )

        return '\n'.join(details) if details else "None"

    def generate_summary_qa(
        self,
        scene_id: str,
        components: List[ScaffoldComponent],
        config: Dict[str, Any]
    ) -> QAPair:
        """
        Generate summary question-answer pair.

        CRITICAL: Uses SAME question template for both Yes and No cases.
        """
        scaffold_spec = self._get_scaffold_spec(config)

        # UNIFORM question template - same for Yes and No
        question = self.q_config.summary_question_template.format(
            scaffold_spec=scaffold_spec
        )

        # Get missing components
        missing_comps = [c for c in components if c.semantic_id == 10]

        # Build answer based on actual state
        if missing_comps:
            missing_details = self._generate_missing_details(missing_comps)
            answer = self.a_config.yes_summary_template.format(
                scaffold_spec=scaffold_spec,
                expected_verticals=config['expected_verticals'],
                expected_horizontals=config['expected_horizontals'],
                expected_platforms=config['expected_platforms'],
                actual_verticals=config['actual_verticals'],
                actual_horizontals=config['actual_horizontals'],
                actual_platforms=config['actual_platforms'],
                num_missing=len(missing_comps),
                missing_details=missing_details
            )
            label = 'Yes'
            bboxes = [
                c.bbox_norm.tolist()
                for c in missing_comps
                if c.bbox_norm is not None
            ]
        else:
            answer = self.a_config.no_summary_template.format(
                scaffold_spec=scaffold_spec,
                expected_verticals=config['expected_verticals'],
                expected_horizontals=config['expected_horizontals'],
                expected_platforms=config['expected_platforms'],
                actual_verticals=config['actual_verticals'],
                actual_horizontals=config['actual_horizontals'],
                actual_platforms=config['actual_platforms']
            )
            label = 'No'
            bboxes = []

        return QAPair(
            question_id=f"{scene_id}_summary",
            point_file=f"{scene_id}.npy",
            question=question,
            answer=answer,
            label=label,
            bboxes=bboxes,
            task_type='missing_detection_summary',
            metadata={'num_missing': len(missing_comps)}
        )

    def generate_floor_qa(
        self,
        scene_id: str,
        components: List[ScaffoldComponent],
        config: Dict[str, Any],
        target_floor: int
    ) -> QAPair:
        """Generate floor-specific question-answer pair."""

        # UNIFORM question template
        question = self.q_config.floor_question_template.format(
            floor_num=target_floor
        )

        # Find missing on this floor
        missing_on_floor = [
            c for c in components
            if c.semantic_id == 10 and
            c.metadata and
            str(c.metadata.get('floor', '')) == str(target_floor)
        ]

        if missing_on_floor:
            details = []
            for comp in missing_on_floor:
                metadata = comp.metadata or {}
                defect_type = metadata.get('defect_type', 'unknown')

                if defect_type == 'missing_horizontal':
                    bay = metadata.get('bay', metadata.get('column', '?'))
                    details.append(
                        f"- Horizontal beam at bay/column {bay}: "
                        f"{self._format_bbox(comp.bbox_norm)}"
                    )
                elif defect_type == 'missing_platform':
                    bay = metadata.get('bay', '?')
                    details.append(
                        f"- Platform at bay {bay}: "
                        f"{self._format_bbox(comp.bbox_norm)}"
                    )

            answer = (
                f"Yes, {len(missing_on_floor)} component(s) are missing on floor {target_floor}:\n"
                + '\n'.join(details)
            )
            label = 'Yes'
            bboxes = [
                c.bbox_norm.tolist()
                for c in missing_on_floor
                if c.bbox_norm is not None
            ]
        else:
            answer = f"No missing components detected on floor {target_floor}. All components are properly installed."
            label = 'No'
            bboxes = []

        return QAPair(
            question_id=f"{scene_id}_floor_{target_floor}",
            point_file=f"{scene_id}.npy",
            question=question,
            answer=answer,
            label=label,
            bboxes=bboxes,
            task_type='missing_detection_floor',
            metadata={'target_floor': target_floor}
        )

    def generate_bay_qa(
        self,
        scene_id: str,
        components: List[ScaffoldComponent],
        config: Dict[str, Any],
        target_bay: int
    ) -> QAPair:
        """Generate bay-specific question-answer pair."""

        # UNIFORM question template
        question = self.q_config.bay_question_template.format(
            bay_num=target_bay
        )

        # Find missing in this bay
        missing_in_bay = [
            c for c in components
            if c.semantic_id == 10 and
            c.metadata and
            str(c.metadata.get('bay', '')) == str(target_bay)
        ]

        if missing_in_bay:
            details = []
            for comp in missing_in_bay:
                metadata = comp.metadata or {}
                defect_type = metadata.get('defect_type', 'unknown')
                floor = metadata.get('floor', '?')

                if defect_type == 'missing_horizontal':
                    details.append(
                        f"- Horizontal beam at floor {floor}: "
                        f"{self._format_bbox(comp.bbox_norm)}"
                    )
                elif defect_type == 'missing_platform':
                    details.append(
                        f"- Platform at floor {floor}: "
                        f"{self._format_bbox(comp.bbox_norm)}"
                    )

            answer = (
                f"Yes, {len(missing_in_bay)} component(s) are missing in bay {target_bay}:\n"
                + '\n'.join(details)
            )
            label = 'Yes'
            bboxes = [
                c.bbox_norm.tolist()
                for c in missing_in_bay
                if c.bbox_norm is not None
            ]
        else:
            answer = f"No missing components detected in bay {target_bay}. All components are properly installed."
            label = 'No'
            bboxes = []

        return QAPair(
            question_id=f"{scene_id}_bay_{target_bay}",
            point_file=f"{scene_id}.npy",
            question=question,
            answer=answer,
            label=label,
            bboxes=bboxes,
            task_type='missing_detection_bay',
            metadata={'target_bay': target_bay}
        )

    def generate_specific_component_qa(
        self,
        scene_id: str,
        component: ScaffoldComponent,
        qa_index: int
    ) -> QAPair:
        """Generate specific component question-answer pair."""

        metadata = component.metadata or {}

        # Determine component type and location
        if component.semantic_id == 0:  # Vertical post (present)
            comp_type = "vertical post"
            col = metadata.get('column', '?')
            row = metadata.get('row', '?')
            location = f"column {col}, row {row}"
            is_missing = False
        elif component.semantic_id == 1:  # Horizontal beam (present)
            comp_type = "horizontal beam"
            floor = metadata.get('floor', '?')
            orientation = metadata.get('orientation', '?')
            if orientation == 'X':
                bay = metadata.get('bay', '?')
                location = f"floor {floor}, bay {bay} (X-direction)"
            else:
                col = metadata.get('column', '?')
                location = f"floor {floor}, column {col} (Y-direction)"
            is_missing = False
        elif component.semantic_id == 3:  # Platform (present)
            comp_type = "platform"
            floor = metadata.get('floor', '?')
            bay = metadata.get('bay', '?')
            location = f"floor {floor}, bay {bay}"
            is_missing = False
        elif component.semantic_id == 10:  # Missing component
            defect_type = metadata.get('defect_type', 'unknown')
            if defect_type == 'missing_vertical':
                comp_type = "vertical post"
                col = metadata.get('column', '?')
                row = metadata.get('row', '?')
                location = f"column {col}, row {row}"
            elif defect_type == 'missing_horizontal':
                comp_type = "horizontal beam"
                floor = metadata.get('floor', '?')
                orientation = metadata.get('orientation', '?')
                if orientation == 'X':
                    bay = metadata.get('bay', '?')
                    location = f"floor {floor}, bay {bay} (X-direction)"
                else:
                    col = metadata.get('column', '?')
                    location = f"floor {floor}, column {col} (Y-direction)"
            else:  # Platform
                comp_type = "platform"
                floor = metadata.get('floor', '?')
                bay = metadata.get('bay', '?')
                location = f"floor {floor}, bay {bay}"
            is_missing = True
        else:
            return None

        # UNIFORM question template
        question = self.q_config.specific_question_template.format(
            component_type=comp_type,
            location=location
        )

        if is_missing:
            answer = (
                f"No, the {comp_type} at {location} is MISSING. "
                f"Expected location: {self._format_bbox(component.bbox_norm)}."
            )
            label = 'No'  # Component is NOT present
            bboxes = [component.bbox_norm.tolist()] if component.bbox_norm is not None else []
        else:
            answer = (
                f"Yes, there is a {comp_type} at {location}. "
                f"Location: {self._format_bbox(component.bbox_norm)}."
            )
            label = 'Yes'  # Component IS present
            bboxes = [component.bbox_norm.tolist()] if component.bbox_norm is not None else []

        return QAPair(
            question_id=f"{scene_id}_specific_{qa_index:03d}",
            point_file=f"{scene_id}.npy",
            question=question,
            answer=answer,
            label=label,
            bboxes=bboxes,
            task_type='missing_detection_specific',
            metadata={'component_type': comp_type, 'location': location, 'is_missing': is_missing}
        )

    def generate_vertical_summary_qa(
        self,
        scene_id: str,
        components: List[ScaffoldComponent],
        config: Dict[str, Any]
    ) -> Optional[QAPair]:
        """Generate vertical posts summary question-answer pair."""

        # UNIFORM question template
        question = self.q_config.vertical_question_template

        # Find missing verticals
        missing_verticals = [
            c for c in components
            if c.semantic_id == 10 and
            c.metadata and
            c.metadata.get('defect_type') == 'missing_vertical'
        ]

        if missing_verticals:
            details = []
            for comp in missing_verticals:
                metadata = comp.metadata or {}
                col = metadata.get('column', '?')
                row = metadata.get('row', '?')
                details.append(
                    f"- Column {col}, row {row}: {self._format_bbox(comp.bbox_norm)}"
                )

            answer = (
                f"Expected: {config['expected_verticals']} vertical posts.\n"
                f"Actual: {config['actual_verticals']} vertical posts.\n"
                f"Missing: {len(missing_verticals)} vertical post(s):\n"
                + '\n'.join(details)
            )
            label = 'Yes'
            bboxes = [
                c.bbox_norm.tolist()
                for c in missing_verticals
                if c.bbox_norm is not None
            ]
        else:
            answer = (
                f"Expected: {config['expected_verticals']} vertical posts.\n"
                f"Actual: {config['actual_verticals']} vertical posts.\n"
                f"No missing vertical posts detected. All vertical posts are properly installed."
            )
            label = 'No'
            bboxes = []

        return QAPair(
            question_id=f"{scene_id}_vertical_summary",
            point_file=f"{scene_id}.npy",
            question=question,
            answer=answer,
            label=label,
            bboxes=bboxes,
            task_type='missing_detection_vertical_summary',
            metadata={'num_missing_verticals': len(missing_verticals)}
        )

    def generate_horizontal_summary_qa(
        self,
        scene_id: str,
        components: List[ScaffoldComponent],
        config: Dict[str, Any]
    ) -> Optional[QAPair]:
        """Generate horizontal beams summary question-answer pair."""

        # UNIFORM question template
        question = self.q_config.horizontal_question_template

        # Find missing horizontals
        missing_horizontals = [
            c for c in components
            if c.semantic_id == 10 and
            c.metadata and
            c.metadata.get('defect_type') == 'missing_horizontal'
        ]

        if missing_horizontals:
            details = []
            for comp in missing_horizontals:
                metadata = comp.metadata or {}
                floor = metadata.get('floor', '?')
                orientation = metadata.get('orientation', '?')
                if orientation == 'X':
                    bay = metadata.get('bay', '?')
                    details.append(
                        f"- Floor {floor}, bay {bay} (X-direction): "
                        f"{self._format_bbox(comp.bbox_norm)}"
                    )
                else:
                    col = metadata.get('column', '?')
                    details.append(
                        f"- Floor {floor}, column {col} (Y-direction): "
                        f"{self._format_bbox(comp.bbox_norm)}"
                    )

            answer = (
                f"Expected: {config['expected_horizontals']} horizontal beams.\n"
                f"Actual: {config['actual_horizontals']} horizontal beams.\n"
                f"Missing: {len(missing_horizontals)} horizontal beam(s):\n"
                + '\n'.join(details)
            )
            label = 'Yes'
            bboxes = [
                c.bbox_norm.tolist()
                for c in missing_horizontals
                if c.bbox_norm is not None
            ]
        else:
            answer = (
                f"Expected: {config['expected_horizontals']} horizontal beams.\n"
                f"Actual: {config['actual_horizontals']} horizontal beams.\n"
                f"No missing horizontal beams detected. All horizontal beams are properly installed."
            )
            label = 'No'
            bboxes = []

        return QAPair(
            question_id=f"{scene_id}_horizontal_summary",
            point_file=f"{scene_id}.npy",
            question=question,
            answer=answer,
            label=label,
            bboxes=bboxes,
            task_type='missing_detection_horizontal_summary',
            metadata={'num_missing_horizontals': len(missing_horizontals)}
        )

    def generate_all_qa_pairs(
        self,
        scene_id: str,
        components: List[ScaffoldComponent],
        config: Dict[str, Any]
    ) -> List[QAPair]:
        """
        Generate all question-answer pairs for a scene.

        This includes:
        - Summary question (always)
        - Floor-specific questions (for floors with OR without missing)
        - Bay-specific questions (for bays with OR without missing)
        - Component-specific questions (balanced Yes/No)
        - Vertical summary question
        - Horizontal summary question
        """
        qa_pairs = []

        # 1. Summary question (always included)
        qa_pairs.append(self.generate_summary_qa(scene_id, components, config))

        # 2. Floor-specific questions (ALL floors, not just those with missing)
        num_floors = config['num_floors']
        for floor in range(num_floors):
            qa_pairs.append(
                self.generate_floor_qa(scene_id, components, config, floor)
            )

        # 3. Bay-specific questions (ALL bays)
        num_bays = config['num_bays']
        for bay in range(num_bays):
            qa_pairs.append(
                self.generate_bay_qa(scene_id, components, config, bay)
            )

        # 4. Component-specific questions (balanced sampling)
        # Get missing components
        missing_comps = [c for c in components if c.semantic_id == 10]

        # Get present components (sample some)
        present_comps = [
            c for c in components
            if c.semantic_id in [0, 1, 3]  # verticals, horizontals, platforms
        ]

        # Generate questions for missing components
        qa_index = 0
        for comp in missing_comps[:3]:  # Max 3 missing component questions
            qa = self.generate_specific_component_qa(scene_id, comp, qa_index)
            if qa:
                qa_pairs.append(qa)
                qa_index += 1

        # Generate equal number of questions for present components (balancing)
        num_present_qs = min(len(present_comps), max(3, len(missing_comps)))
        sampled_present = random.sample(present_comps, min(num_present_qs, len(present_comps)))

        for comp in sampled_present:
            qa = self.generate_specific_component_qa(scene_id, comp, qa_index)
            if qa:
                qa_pairs.append(qa)
                qa_index += 1

        # 5. Vertical summary (if there are verticals)
        vertical_qa = self.generate_vertical_summary_qa(scene_id, components, config)
        if vertical_qa:
            qa_pairs.append(vertical_qa)

        # 6. Horizontal summary
        horizontal_qa = self.generate_horizontal_summary_qa(scene_id, components, config)
        if horizontal_qa:
            qa_pairs.append(horizontal_qa)

        return qa_pairs


class CaptionGenerator:
    """
    Generator for Stage 1 caption data.

    Stage 1 uses simple descriptive captions for feature alignment,
    not the complex QA format of Stage 2.
    """

    def __init__(self, config: ScaffoldConfig):
        """Initialize with configuration."""
        self.config = config
        self.templates = config.stage1.caption_templates
        self.captions_per_scene = config.stage1.captions_per_scene

    def generate_captions(
        self,
        scene_id: str,
        components: List[ScaffoldComponent],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate caption entries for Stage 1 training."""
        captions = []

        # Calculate component counts
        num_verticals = len([c for c in components if c.semantic_id == 0])
        num_horizontals = len([c for c in components if c.semantic_id == 1])
        num_platforms = len([c for c in components if c.semantic_id == 3])
        total_components = num_verticals + num_horizontals + num_platforms

        # Template variables
        template_vars = {
            'num_bays': config['num_bays'],
            'num_floors': config['num_floors'],
            'num_verticals': num_verticals,
            'num_horizontals': num_horizontals,
            'num_platforms': num_platforms,
            'width': config.get('total_width', config['num_bays'] * config['bay_width']),
            'depth': config['depth'],
            'height': config.get('total_height', config['num_floors'] * config['floor_height']),
            'bay_width': config['bay_width'],
            'safety_status': config['safety_status'],
            'total_components': total_components
        }

        # Select random templates
        selected_templates = random.sample(
            self.templates,
            min(self.captions_per_scene, len(self.templates))
        )

        for i, template in enumerate(selected_templates):
            try:
                caption_text = template.format(**template_vars)
            except KeyError:
                caption_text = f"A {config['num_bays']}-bay, {config['num_floors']}-floor scaffold structure."

            captions.append({
                'id': f"{scene_id}_caption_{i}",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {'from': 'human', 'value': '<point>'},
                    {'from': 'gpt', 'value': caption_text}
                ]
            })

        return captions
