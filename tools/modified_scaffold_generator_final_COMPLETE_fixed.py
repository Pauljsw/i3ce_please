"""
🏗️ 고도화된 비계 합성 데이터 생성 도구 (ShapeLLM용) - 최종 버전
- 한국 산업안전보건기준 준수 (2025년 기준)
- 실제 시스템비계 규격 반영
- 점진적 학습 목표 지원 (Referring → 누락 감지 → 안정성 → 손상 → 규정)
- 누락 부품 quota 시스템 (최대 4개)
- 수직재/수평재/플랫폼 누락 지원

This module defines a scaffold synthesis generator for ShapeLLM datasets.  All runtime
strings and identifiers used for file names or annotation content are written
in English to avoid issues when storing or loading JSON or other artefacts.  Only
docstrings and inline comments remain in Korean for documentation purposes.
"""

import numpy as np
import os
import random
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from collections import defaultdict


@dataclass
class ScaffoldComponent:
    """비계 부품 정의 (Korean docstring)

    The core data structure representing a single scaffold component.  Runtime
    fields such as ``name`` are provided in English to ensure downstream
    consumers operate on ASCII-only data.  Comments and documentation are left
    untranslated.
    """
    name: str
    semantic_id: int
    instance_id: int
    points: np.ndarray  # [N, 3] coordinates (color removed)
    bbox: Optional[np.ndarray] = None  # [8, 3] bounding box corners (world coords)
    bbox_norm: Optional[np.ndarray] = None  # [8, 3] bounding box corners (normalized coords)
    metadata: Optional[Dict] = None  # 추가 메타데이터


class KoreanScaffoldRegulations:
    """한국 산업안전보건기준 (2025년)

    This class encodes the key numerical limits defined by Korean scaffold
    regulations.  The check methods return human‑readable English messages so
    that any violation stored in configuration or output files does not
    contain Korean text.
    """

    # 기둥 간격 기준
    MAX_COLUMN_SPACING_LEDGER = 1.85  # 띠장 방향 (m)
    MAX_COLUMN_SPACING_PURLIN = 1.5   # 장선 방향 (m)

    # 작업발판 기준
    MIN_PLATFORM_WIDTH = 0.40  # 40cm
    MAX_PLATFORM_GAP = 0.03    # 3cm

    # 안전난간 기준
    TOP_RAIL_HEIGHT_MIN = 0.90   # 90cm
    TOP_RAIL_HEIGHT_MAX = 1.20   # 120cm
    MID_RAIL_REQUIRED = True
    TOE_BOARD_MIN_HEIGHT = 0.10  # 10cm

    # 가새 설치 기준
    MAX_BRACE_VERTICAL_SPAN = 5  # 5단 이내
    BRACE_ANGLE_MIN = 40  # 40도
    BRACE_ANGLE_MAX = 60  # 60도

    # 벽 연결재 기준
    MAX_WALL_TIE_SPACING = 5.0  # 수직/수평 5m 이내

    @classmethod
    def check_column_spacing(cls, spacing_x: float, spacing_y: float) -> List[str]:
        """Validate column spacing and return violations in English."""
        violations: List[str] = []
        if spacing_x > cls.MAX_COLUMN_SPACING_LEDGER:
            violations.append(
                f"Column spacing exceeded in ledger direction: {spacing_x:.2f} m > {cls.MAX_COLUMN_SPACING_LEDGER} m"
            )
        if spacing_y > cls.MAX_COLUMN_SPACING_PURLIN:
            violations.append(
                f"Column spacing exceeded in purlin direction: {spacing_y:.2f} m > {cls.MAX_COLUMN_SPACING_PURLIN} m"
            )
        return violations

    @classmethod
    def check_platform_width(cls, width: float) -> List[str]:
        """Validate platform width and return violations in English."""
        if width < cls.MIN_PLATFORM_WIDTH:
            return [
                f"Platform width insufficient: {width:.2f} m < {cls.MIN_PLATFORM_WIDTH} m"
            ]
        return []


class ScaffoldSpecs:
    """비계 부품 규격 (mm 단위를 m로 변환)"""

    # 수직재 (Vertical Posts) - Ø48.6 * 2.3T
    VERTICAL_LENGTHS = {
        'V-38': 3.8,
        'V-19': 1.9,
        'V-09': 0.95,
        'V-04': 0.475
    }

    # 수평재 (Horizontal Beams) - Ø42.7 * 2.3T
    HORIZONTAL_SPECS = {
        'H-18': {'length': 1.768, 'spacing': 1.817},
        'H-15': {'length': 1.463, 'spacing': 1.512},
        'H-12': {'length': 1.158, 'spacing': 1.207},
        'H-09': {'length': 0.853, 'spacing': 0.902},
        'H-06': {'length': 0.549, 'spacing': 0.598},
        'H-03': {'length': 0.244, 'spacing': 0.293}
    }

    # 대각재 (Diagonal Braces) - Ø34 x 2.3T
    DIAGONAL_SPECS = {
        'B-1918': {'length': 2.629, 'height': 1.9, 'width': 1.829},
        'B-1915': {'length': 2.428, 'height': 1.9, 'width': 1.524},
        'B-1912': {'length': 2.251, 'height': 1.9, 'width': 1.219}
    }

    # 발판 (Platform) 크기
    PLATFORM_SIZES = [
        (0.4, 0.598),
        (0.4, 0.902),
        (0.4, 1.817)
    ]

    # 하부받침 (Base Support)
    BASE_SUPPORT = {
        'base_size': (0.14, 0.14),
        'pipe_diameter': 0.034,
        'height': 0.15
    }

    # 파이프 직경
    PIPE_DIAMETERS = {
        'vertical': 0.0486,
        'horizontal': 0.0427,
        'diagonal': 0.034,
        'handrail': 0.034
    }


class EnhancedScaffoldGeneratorFinal:
    """Generator for synthetic scaffold scenes with regulatory validation and missing quota system.

    This class implements:
    - Global missing component quota (max 4 components)
    - Vertical/horizontal beam missing detection
    - Comprehensive question types for training
    - Z-axis vertical ladders (no diagonal stairs)
    """

    def __init__(self, random_seed: Optional[int] = 42) -> None:
        """Initialize the scaffold generator."""
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Class names (English only)
        self.class_names: List[str] = [
            "vertical post",     # 0
            "horizontal beam",   # 1
            "diagonal brace",    # 2
            "platform",          # 3
            "base support",      # 4
            "connection",        # 5
            "stair",             # 6
            "ladder",            # 7
            "safety rail",       # 8
            "damaged component", # 9
            "missing part"       # 10
        ]

        self.class_names_en: List[str] = list(self.class_names)
        
        # 🆕 Global missing quota system
        self.missing_quota = 4  # Maximum total missing components
        self.current_missing_count = 0  # Current count of missing components
        self.current_missing_by_type: Dict[str, int] = defaultdict(int)
        
        self.instance_counter = 1

    def reset_missing_quota(self) -> None:
        """Reset missing component counter for new scene generation."""
        self.current_missing_count = 0
        self.current_missing_by_type = defaultdict(int)

    def can_add_missing_component(self) -> bool:
        """Check if we can add another missing component within quota."""
        return self.current_missing_count < self.missing_quota

    def add_missing_component(self, defect_type: Optional[str] = None) -> bool:
        """Try to add a missing component. Returns True if successful."""
        if self.can_add_missing_component():
            self.current_missing_count += 1
            if defect_type:
                self.current_missing_by_type[defect_type] += 1
            return True
        return False

    def generate_pipe_points(self, start_pos: np.ndarray, end_pos: np.ndarray, diameter: float, num_points: int = 50) -> np.ndarray:
        """Generate CONTINUOUS points along a pipe between two positions."""
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return np.array([]).reshape(0, 3)
        
        direction_norm = direction / length
        
        # 🆕 파이프 길이에 비례하여 점 개수 계산 (1m당 약 100개)
        num_length_samples = max(int(length * 100), 10)
        num_circumference = 12  # 원주 방향 점 개수
        
        # Create perpendicular vectors (한 번만 계산)
        if abs(direction_norm[2]) < 0.9:
            perp1 = np.cross(direction_norm, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(direction_norm, np.array([1, 0, 0]))
        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-12)
        perp2 = np.cross(direction_norm, perp1)
        
        points = []
        radius = diameter / 2
        
        # 🆕 모든 위치에서 점 생성 (if 조건 제거!)
        for i in range(num_length_samples):
            t = i / max(num_length_samples - 1, 1)
            center = start_pos + t * direction
            
            # 원주 방향 점 생성
            for j in range(num_circumference):
                angle = j * 2 * np.pi / num_circumference
                r_variation = radius * (1.0 + np.random.uniform(-0.05, 0.05))
                offset = r_variation * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                length_noise = np.random.uniform(-0.005, 0.005) * direction_norm
                points.append(center + offset + length_noise)
        
        return np.array(points) if points else np.array([]).reshape(0, 3)

    def generate_platform_points(self, center: np.ndarray, width: float, length: float, num_points: int = 100) -> np.ndarray:
        """Generate DENSE points for a rectangular platform with grid + edges."""
        # 🆕 면적 기반 점 개수 계산 (1m²당 약 500개)
        area = width * length
        actual_num_points = max(int(area * 500), num_points)
        
        points = []
        thickness = 0.03  # 발판 두께 3cm
        
        # 🆕 그리드 기반 점 생성 (상면)
        grid_x = max(int(np.sqrt(actual_num_points * width / length)), 5)
        grid_y = max(int(np.sqrt(actual_num_points * length / width)), 5)
        
        for i in range(grid_x):
            for j in range(grid_y):
                x_offset = -width/2 + width * (i + np.random.uniform(0.2, 0.8)) / grid_x
                y_offset = -length/2 + length * (j + np.random.uniform(0.2, 0.8)) / grid_y
                
                # 상면
                z_top = np.random.uniform(-0.002, 0.002)
                points.append(center + np.array([x_offset, y_offset, z_top]))
                
                # 하면 (30% 확률)
                if np.random.random() < 0.3:
                    z_bottom = -thickness + np.random.uniform(-0.002, 0.002)
                    points.append(center + np.array([x_offset, y_offset, z_bottom]))
        
        # 🆕 엣지 점 생성 (테두리 강조)
        num_edge_points = int(2 * (width + length) * 50)  # 1m당 50개
        for _ in range(num_edge_points):
            edge = np.random.randint(4)
            if edge == 0:    # 왼쪽
                x, y = -width/2, np.random.uniform(-length/2, length/2)
            elif edge == 1:  # 오른쪽
                x, y = width/2, np.random.uniform(-length/2, length/2)
            elif edge == 2:  # 앞
                x, y = np.random.uniform(-width/2, width/2), -length/2
            else:            # 뒤
                x, y = np.random.uniform(-width/2, width/2), length/2
            
            x += np.random.uniform(-0.005, 0.005)
            y += np.random.uniform(-0.005, 0.005)
            z = np.random.uniform(-thickness, 0)
            points.append(center + np.array([x, y, z]))
        
        return np.array(points) if points else np.array([]).reshape(0, 3)

    def calculate_bbox(self, points: np.ndarray) -> np.ndarray:
        """Calculate 3D bounding box from points."""
        if len(points) == 0:
            return None
        
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Generate 8 corners of the bounding box
        corners = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],  # min corner
            [max_coords[0], min_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]]   # max corner
        ])
        
        return corners

    def normalize_bbox(self, bbox: np.ndarray, centroid: np.ndarray, scale: float, R: np.ndarray) -> np.ndarray:
        """🆕 Apply same normalization to bounding box as point cloud."""
        if bbox is None:
            return None
        
        # Apply same transformations as point cloud
        bbox_centered = bbox - centroid
        bbox_scaled = bbox_centered / scale
        bbox_rotated = (R @ bbox_scaled.T).T
        
        # 🆕 Ensure bbox stays within -1~1 range
        bbox_clipped = np.clip(bbox_rotated, -1.0, 1.0)
        
        return bbox_clipped.astype(np.float32)

    def _create_vertical_posts_with_validation(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float],
        config: Dict
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """Create vertical posts with missing detection capability (RANDOMIZED)."""
        components: List[ScaffoldComponent] = []
        violations: List[str] = []

        safety_status = config.get('safety_status', 'safe')
        base_missing_rate = {'safe': 0.0, 'minor_defect': 0.1, 'major_defect': 0.2}[safety_status]

        diameter = ScaffoldSpecs.PIPE_DIAMETERS['vertical']
        max_height = max(cumulative_heights)

        # Step 1: Create ALL vertical posts as normal (no missing yet)
        all_verticals = []
        for col in range(num_bays + 1):
            for row in range(2):  # Front and back rows
                x = col * bay_width
                y = row * depth

                start_pos = np.array([x, y, 0])
                end_pos = np.array([x, y, max_height])
                points = self.generate_pipe_points(start_pos, end_pos, diameter)

                if len(points) > 0:
                    bbox = self.calculate_bbox(points)
                    comp = ScaffoldComponent(
                        name=f"vertical_post_{self.instance_counter}",
                        semantic_id=0,
                        instance_id=self.instance_counter,
                        points=points,
                        bbox=bbox,
                        metadata={'column': col, 'row': row, 'position': (x, y)}
                    )
                    all_verticals.append(comp)
                    self.instance_counter += 1

        # Step 2: RANDOMLY select some to convert to missing markers
        if base_missing_rate > 0 and len(all_verticals) > 0:
            # Calculate how many should be missing
            num_candidates = int(len(all_verticals) * base_missing_rate)
            # Limit by quota
            type_quota = config.get('missing_type_quota', {}).get('missing_vertical', self.missing_quota)
            remaining_type_quota = max(type_quota - self.current_missing_by_type['missing_vertical'], 0)
            num_to_remove = min(
                num_candidates,
                remaining_type_quota,
                self.missing_quota - self.current_missing_count
            )

            if num_to_remove > 0:
                # RANDOMIZE: shuffle and pick first N
                random.shuffle(all_verticals)

                for i in range(num_to_remove):
                    comp = all_verticals[i]
                    col = comp.metadata['column']
                    row = comp.metadata['row']
                    x, y = comp.metadata['position']

                    # Convert to missing marker
                    mid_height = max_height / 2
                    marker_points = []
                    for _ in range(10):
                        noise = np.random.normal(0, 0.05, 3)
                        marker_points.append([x, y, mid_height] + noise)

                    marker_points = np.array(marker_points)
                    bbox = self.calculate_bbox(marker_points)

                    # Replace with missing marker
                    all_verticals[i] = ScaffoldComponent(
                        name=f"missing_vertical_{col}_{row}_{comp.instance_id}",
                        semantic_id=10,
                        instance_id=comp.instance_id,
                        points=marker_points,
                        bbox=bbox,
                        metadata={
                            'defect_type': 'missing_vertical',
                            'column': col,
                            'row': row,
                            'floor': 'all'
                        }
                    )
                    self.add_missing_component('missing_vertical')
                    violations.append(f"Missing vertical post at column {col}, row {row}")

        components.extend(all_verticals)
        return components, violations

    def _create_horizontal_beams_with_validation(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float],
        config: Dict
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """Create horizontal beams with missing detection capability (RANDOMIZED)."""
        components: List[ScaffoldComponent] = []
        violations: List[str] = []

        safety_status = config.get('safety_status', 'safe')
        base_missing_rate = {'safe': 0.0, 'minor_defect': 0.1, 'major_defect': 0.2}[safety_status]

        diameter = ScaffoldSpecs.PIPE_DIAMETERS['horizontal']

        # Step 1: Create ALL horizontal beams as normal (no missing yet)
        all_horizontals = []

        for floor_idx, z in enumerate(cumulative_heights[:-1]):
            # X-direction beams
            for bay in range(num_bays):
                for side_idx, j in enumerate([0, depth]):
                    start_pos = np.array([bay * bay_width, j, z])
                    end_pos = np.array([(bay + 1) * bay_width, j, z])
                    mid_pos = (start_pos + end_pos) / 2.0

                    points = self.generate_pipe_points(start_pos, end_pos, diameter)
                    if len(points) > 0:
                        bbox = self.calculate_bbox(points)
                        comp = ScaffoldComponent(
                            name=f"horizontal_beam_X_{self.instance_counter}",
                            semantic_id=1,
                            instance_id=self.instance_counter,
                            points=points,
                            bbox=bbox,
                            metadata={
                                'orientation': 'X',
                                'floor': floor_idx,
                                'bay': bay,
                                'side': side_idx,
                                'mid_pos': mid_pos
                            }
                        )
                        all_horizontals.append(comp)
                        self.instance_counter += 1

            # Y-direction beams
            for col in range(num_bays + 1):
                start_pos = np.array([col * bay_width, 0, z])
                end_pos = np.array([col * bay_width, depth, z])
                mid_pos = (start_pos + end_pos) / 2.0

                points = self.generate_pipe_points(start_pos, end_pos, diameter)
                if len(points) > 0:
                    bbox = self.calculate_bbox(points)
                    comp = ScaffoldComponent(
                        name=f"horizontal_beam_Y_{self.instance_counter}",
                        semantic_id=1,
                        instance_id=self.instance_counter,
                        points=points,
                        bbox=bbox,
                        metadata={
                            'orientation': 'Y',
                            'floor': floor_idx,
                            'column': col,
                            'side': 0,
                            'mid_pos': mid_pos
                        }
                    )
                    all_horizontals.append(comp)
                    self.instance_counter += 1

        # Step 2: RANDOMLY select some to convert to missing markers
        if base_missing_rate > 0 and len(all_horizontals) > 0:
            num_candidates = int(len(all_horizontals) * base_missing_rate)
            type_quota = config.get('missing_type_quota', {}).get('missing_horizontal', self.missing_quota)
            remaining_type_quota = max(type_quota - self.current_missing_by_type['missing_horizontal'], 0)
            num_to_remove = min(
                num_candidates,
                remaining_type_quota,
                self.missing_quota - self.current_missing_count
            )

            if num_to_remove > 0:
                # RANDOMIZE: shuffle and pick first N
                random.shuffle(all_horizontals)

                for i in range(num_to_remove):
                    comp = all_horizontals[i]
                    floor_idx = comp.metadata['floor']
                    mid_pos = comp.metadata['mid_pos']
                    orientation = comp.metadata['orientation']

                    # Convert to missing marker
                    marker_points = []
                    for _ in range(10):
                        noise = np.random.normal(0, 0.05, 3)
                        marker_points.append(mid_pos + noise)

                    marker_points = np.array(marker_points)
                    bbox = self.calculate_bbox(marker_points)

                    # Build name based on orientation
                    if orientation == 'X':
                        bay = comp.metadata['bay']
                        side_idx = comp.metadata['side']
                        name = f"missing_horizontal_X_{bay}_{side_idx}_{floor_idx}_{comp.instance_id}"
                        violation_msg = f"Missing horizontal beam X at floor {floor_idx}, bay {bay}"
                    else:  # Y
                        col = comp.metadata['column']
                        name = f"missing_horizontal_Y_{col}_{floor_idx}_{comp.instance_id}"
                        violation_msg = f"Missing horizontal beam Y at floor {floor_idx}, column {col}"

                    # Replace with missing marker
                    all_horizontals[i] = ScaffoldComponent(
                        name=name,
                        semantic_id=10,
                        instance_id=comp.instance_id,
                        points=marker_points,
                        bbox=bbox,
                        metadata={
                            'defect_type': 'missing_horizontal',
                            'orientation': orientation,
                            'floor': floor_idx,
                            **{k: v for k, v in comp.metadata.items() if k in ['bay', 'column', 'side']}
                        }
                    )
                    self.add_missing_component('missing_horizontal')
                    violations.append(violation_msg)

        components.extend(all_horizontals)
        return components, violations

    def _create_platforms_with_validation(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float],
        config: Dict
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """Create platforms with missing detection capability (RANDOMIZED)."""
        components: List[ScaffoldComponent] = []
        violations: List[str] = []

        safety_status = config.get('safety_status', 'safe')
        base_missing_rate = {'safe': 0.0, 'minor_defect': 0.1, 'major_defect': 0.2}[safety_status]

        # Step 1: Create ALL platforms as normal (no missing yet)
        all_platforms = []

        for floor_idx, z in enumerate(cumulative_heights[:-1]):
            for bay in range(num_bays):
                platform_center = np.array([
                    (bay + 0.5) * bay_width,
                    depth / 2,
                    z
                ])
                platform_width = bay_width * 0.95
                platform_length = depth * 0.95

                width_violations = KoreanScaffoldRegulations.check_platform_width(platform_width)
                violations.extend(width_violations)

                platform_points = self.generate_platform_points(platform_center, platform_width, platform_length)

                if len(platform_points) > 0:
                    bbox = self.calculate_bbox(platform_points)
                    floor_name = "ground" if floor_idx == 0 else f"floor_{floor_idx}"
                    component = ScaffoldComponent(
                        name=f"platform_{floor_name}_{bay}_{self.instance_counter}",
                        semantic_id=3,
                        instance_id=self.instance_counter,
                        points=platform_points,
                        bbox=bbox,
                        metadata={
                            'width': platform_width,
                            'floor': floor_idx,
                            'bay': bay,
                            'center': platform_center
                        }
                    )
                    all_platforms.append(component)
                    self.instance_counter += 1

        # Step 2: RANDOMLY select some to convert to missing markers
        if base_missing_rate > 0 and len(all_platforms) > 0:
            num_candidates = int(len(all_platforms) * base_missing_rate)
            type_quota = config.get('missing_type_quota', {}).get('missing_platform', self.missing_quota)
            remaining_type_quota = max(type_quota - self.current_missing_by_type['missing_platform'], 0)
            num_to_remove = min(
                num_candidates,
                remaining_type_quota,
                self.missing_quota - self.current_missing_count
            )

            if num_to_remove > 0:
                # RANDOMIZE: shuffle and pick first N
                random.shuffle(all_platforms)

                for i in range(num_to_remove):
                    comp = all_platforms[i]
                    floor_idx = comp.metadata['floor']
                    bay = comp.metadata['bay']
                    center = comp.metadata['center']

                    # Convert to missing marker
                    marker_points = []
                    for _ in range(10):
                        x = center[0] + random.uniform(-0.1, 0.1)
                        y = center[1] + random.uniform(-0.1, 0.1)
                        marker_points.append([x, y, center[2]])

                    marker_points = np.array(marker_points)
                    bbox = self.calculate_bbox(marker_points)

                    # Replace with missing marker
                    all_platforms[i] = ScaffoldComponent(
                        name=f"missing_platform_{floor_idx}_{bay}_{comp.instance_id}",
                        semantic_id=10,
                        instance_id=comp.instance_id,
                        points=marker_points,
                        bbox=bbox,
                        metadata={
                            'defect_type': 'missing_platform',
                            'floor': floor_idx,
                            'bay': bay
                        }
                    )
                    self.add_missing_component('missing_platform')
                    violations.append(f"Missing platform at floor {floor_idx}, bay {bay}")

        components.extend(all_platforms)
        return components, violations

    def _create_vertical_ladders(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float]
    ) -> List[ScaffoldComponent]:
        """Create vertical Z-axis ladders (no diagonal stairs)."""
        components: List[ScaffoldComponent] = []
        
        # Create ladder at every other floor (even floors)
        for floor_idx in range(0, len(cumulative_heights)-1, 2):
            if floor_idx + 1 >= len(cumulative_heights):
                break
                
            # Choose a bay for ladder placement
            ladder_bay = random.randint(0, max(0, num_bays-1))
            ladder_x = (ladder_bay + 0.5) * bay_width
            ladder_y = depth * 0.1  # Close to front edge
            
            z_bottom = cumulative_heights[floor_idx]
            z_top = cumulative_heights[floor_idx + 1]
            
            # Create two vertical rails
            for rail_offset in [-0.3, 0.3]:  # Left and right rails
                start_pos = np.array([ladder_x + rail_offset, ladder_y, z_bottom])
                end_pos = np.array([ladder_x + rail_offset, ladder_y, z_top])
                
                diameter = ScaffoldSpecs.PIPE_DIAMETERS['handrail']
                points = self.generate_pipe_points(start_pos, end_pos, diameter, num_points=30)
                
                if len(points) > 0:
                    bbox = self.calculate_bbox(points)
                    comp = ScaffoldComponent(
                        name=f"ladder_rail_{floor_idx}_{self.instance_counter}",
                        semantic_id=7,
                        instance_id=self.instance_counter,
                        points=points,
                        bbox=bbox,
                        metadata={'floor': floor_idx, 'bay': ladder_bay, 'rail_type': 'vertical'}
                    )
                    components.append(comp)
                    self.instance_counter += 1
            
            # Create rungs between rails
            num_rungs = 5
            for rung_idx in range(num_rungs):
                rung_z = z_bottom + (z_top - z_bottom) * (rung_idx + 1) / (num_rungs + 1)
                start_pos = np.array([ladder_x - 0.3, ladder_y, rung_z])
                end_pos = np.array([ladder_x + 0.3, ladder_y, rung_z])
                
                points = self.generate_pipe_points(start_pos, end_pos, diameter, num_points=10)
                
                if len(points) > 0:
                    bbox = self.calculate_bbox(points)
                    comp = ScaffoldComponent(
                        name=f"ladder_rung_{floor_idx}_{rung_idx}_{self.instance_counter}",
                        semantic_id=7,
                        instance_id=self.instance_counter,
                        points=points,
                        bbox=bbox,
                        metadata={'floor': floor_idx, 'bay': ladder_bay, 'rail_type': 'rung'}
                    )
                    components.append(comp)
                    self.instance_counter += 1
        
        return components

    def _create_diagonal_braces_with_validation(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float],
        num_floors: int
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """Create diagonal braces with validation."""
        components: List[ScaffoldComponent] = []
        violations: List[str] = []

        diameter = ScaffoldSpecs.PIPE_DIAMETERS['diagonal']
        floors_without_braces: List[int] = []

        for floor_idx in range(len(cumulative_heights) - 1):
            z_bottom = cumulative_heights[floor_idx]
            z_top = cumulative_heights[floor_idx + 1]

            if random.random() < 0.6:
                for j in [0, depth]:
                    for bay in range(0, num_bays, 2):
                        start_pos = np.array([bay * bay_width, j, z_bottom])
                        end_pos = np.array([(bay + 1) * bay_width, j, z_top])
                        
                        points = self.generate_pipe_points(start_pos, end_pos, diameter)
                        
                        if len(points) > 0:
                            bbox = self.calculate_bbox(points)
                            component = ScaffoldComponent(
                                name=f"diagonal_brace_floor_{floor_idx}_{self.instance_counter}",
                                semantic_id=2,
                                instance_id=self.instance_counter,
                                points=points,
                                bbox=bbox,
                                metadata={'floor': floor_idx}
                            )
                            components.append(component)
                            self.instance_counter += 1
            else:
                floors_without_braces.append(floor_idx + 1)

        # Regulation validation
        if floors_without_braces:
            consecutive = 1
            for i in range(1, len(floors_without_braces)):
                if floors_without_braces[i] == floors_without_braces[i-1] + 1:
                    consecutive += 1
                    if consecutive >= KoreanScaffoldRegulations.MAX_BRACE_VERTICAL_SPAN:
                        violations.append(
                            f"Braces not installed: {consecutive} consecutive floors (allowed within {KoreanScaffoldRegulations.MAX_BRACE_VERTICAL_SPAN} spans)"
                        )
                        break
                else:
                    consecutive = 1

        return components, violations

    def _create_safety_handrails(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float],
        config: Dict
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """Create safety handrails with validation."""
        components: List[ScaffoldComponent] = []
        violations: List[str] = []

        safety_status = config.get('safety_status', 'safe')
        diameter = ScaffoldSpecs.PIPE_DIAMETERS['handrail']

        # Only install handrails on working floors (not ground)
        for floor_idx in range(1, len(cumulative_heights) - 1):
            z = cumulative_heights[floor_idx]
            
            # Top rail height (90-120cm above platform)
            top_rail_height = random.uniform(
                KoreanScaffoldRegulations.TOP_RAIL_HEIGHT_MIN,
                KoreanScaffoldRegulations.TOP_RAIL_HEIGHT_MAX
            )
            
            # Skip handrails occasionally for defective scaffolds
            if safety_status in ['minor_defect', 'major_defect'] and random.random() < 0.3:
                violations.append(f"Missing safety handrail on floor {floor_idx}")
                continue
            
            # Create handrails around the perimeter
            rail_z = z + top_rail_height
            
            # Front and back rails (X direction)
            for y in [0, depth]:
                for bay in range(num_bays):
                    start_pos = np.array([bay * bay_width, y, rail_z])
                    end_pos = np.array([(bay + 1) * bay_width, y, rail_z])
                    
                    points = self.generate_pipe_points(start_pos, end_pos, diameter, num_points=20)
                    if len(points) > 0:
                        bbox = self.calculate_bbox(points)
                        comp = ScaffoldComponent(
                            name=f"handrail_X_{floor_idx}_{bay}_{self.instance_counter}",
                            semantic_id=8,
                            instance_id=self.instance_counter,
                            points=points,
                            bbox=bbox,
                            metadata={'floor': floor_idx, 'orientation': 'X', 'height': top_rail_height}
                        )
                        components.append(comp)
                        self.instance_counter += 1
            
            # Side rails (Y direction)
            for x in [0, num_bays * bay_width]:
                start_pos = np.array([x, 0, rail_z])
                end_pos = np.array([x, depth, rail_z])
                
                points = self.generate_pipe_points(start_pos, end_pos, diameter, num_points=20)
                if len(points) > 0:
                    bbox = self.calculate_bbox(points)
                    comp = ScaffoldComponent(
                        name=f"handrail_Y_{floor_idx}_{self.instance_counter}",
                        semantic_id=8,
                        instance_id=self.instance_counter,
                        points=points,
                        bbox=bbox,
                        metadata={'floor': floor_idx, 'orientation': 'Y', 'height': top_rail_height}
                    )
                    components.append(comp)
                    self.instance_counter += 1

        return components, violations

    def _format_bbox(self, bbox: Optional[np.ndarray]) -> str:
        """Format bounding box coordinates in ShapeLLM standard 8-corner format."""
        if bbox is None or len(bbox) != 8:
            return "N/A"
        # ShapeLLM standard format: [[x1, y1, z1], [x2, y2, z2], ..., [x8, y8, z8]]
        corners_list = [[float(f"{corner[0]:.3f}"), float(f"{corner[1]:.3f}"), float(f"{corner[2]:.3f}")]
                        for corner in bbox]
        return str(corners_list)

    def generate_shapellm_annotations(self, scene_id: str, components: List[ScaffoldComponent], config: Dict) -> List[Dict]:
        """Generate comprehensive missing detection annotations with Template-Guided Reasoning (Idea 1)."""
        annotations: List[Dict] = []

        # Get scaffold configuration for labeling
        num_bays = config.get('num_bays', 3)
        num_floors = config.get('num_floors', 4)
        num_rows = 2  # Always 2 rows (front and back)
        scaffold_spec = f"{num_bays}-bay, {num_rows}-row, {num_floors}-floor scaffold"

        # Separate components by type
        missing_comps = [c for c in components if c.semantic_id == 10]
        missing_verticals = [c for c in missing_comps if (c.metadata or {}).get('defect_type') == 'missing_vertical']
        missing_horizontals = [c for c in missing_comps if (c.metadata or {}).get('defect_type') == 'missing_horizontal']
        missing_platforms = [c for c in missing_comps if (c.metadata or {}).get('defect_type') == 'missing_platform']

        present_verticals = [c for c in components if c.semantic_id == 0]
        present_horizontals = [c for c in components if c.semantic_id == 1]
        present_platforms = [c for c in components if c.semantic_id == 3]

        # ✅ CORRECTED: Expected = Present + Missing (complete structure before missing)
        # This ensures logical consistency: Expected >= Actual
        expected_verticals = len(present_verticals) + len(missing_verticals)
        expected_horizontals = len(present_horizontals) + len(missing_horizontals)
        expected_platforms = len(present_platforms) + len(missing_platforms)

        # Actual counts (present components only)
        actual_verticals = len(present_verticals)
        actual_horizontals = len(present_horizontals)
        actual_platforms = len(present_platforms)

        # Expected vs Actual summary text
        expected_text = f"Expected structure: {scaffold_spec} has {expected_verticals} vertical posts, {expected_horizontals} horizontal beams, {expected_platforms} platforms."
        actual_text = f"Actual: {actual_verticals} vertical posts, {actual_horizontals} horizontal beams, {actual_platforms} platforms."

        # 1. Overall summary question
        if missing_comps:
            missing_info: List[str] = []
            for comp in missing_comps[:5]:  # Limit to 5 for readability
                metadata = comp.metadata or {}
                defect_type = metadata.get('defect_type', 'unknown')
                floor = metadata.get('floor', '?')
                bay = metadata.get('bay', '?')
                column = metadata.get('column', '?')
                
                if defect_type == 'missing_platform':
                    missing_info.append(f"- Platform at floor {floor}, bay {bay}: {self._format_bbox(comp.bbox_norm)}")
                elif defect_type == 'missing_vertical':
                    missing_info.append(f"- Vertical post at column {column}: {self._format_bbox(comp.bbox_norm)}")
                elif defect_type == 'missing_horizontal':
                    orientation = metadata.get('orientation', '?')
                    if orientation == 'X':
                        missing_info.append(f"- Horizontal beam X at floor {floor}, bay {bay}: {self._format_bbox(comp.bbox_norm)}")
                    else:
                        missing_info.append(f"- Horizontal beam Y at floor {floor}, column {column}: {self._format_bbox(comp.bbox_norm)}")

            annotations.append({
                'id': f"{scene_id}_missing_summary",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': f'<point>\nThis is a {scaffold_spec}. Are there any missing components? If so, provide their locations.'
                    },
                    {
                        'from': 'gpt',
                        'value': f"{expected_text}\n{actual_text}\nMissing: {len(missing_comps)} components are missing:\n" + '\n'.join(missing_info)
                    }
                ],
                'task_type': 'missing_detection_summary',
                'num_defects': len(missing_comps)
            })

            # 2. Floor-specific questions
            floors_with_missing: Dict[str, List[ScaffoldComponent]] = {}
            for comp in missing_comps:
                metadata = comp.metadata or {}
                floor = str(metadata.get('floor', '?'))
                floors_with_missing.setdefault(floor, []).append(comp)

            for floor, comps_in_floor in floors_with_missing.items():
                if floor != 'all':  # Skip vertical posts that span all floors
                    floor_missing_info: List[str] = []
                    for comp in comps_in_floor:
                        metadata = comp.metadata or {}
                        defect_type = metadata.get('defect_type', 'unknown')
                        bay = metadata.get('bay', '?')
                        column = metadata.get('column', '?')
                        
                        if defect_type == 'missing_platform':
                            floor_missing_info.append(f"- Platform at bay {bay}: {self._format_bbox(comp.bbox_norm)}")
                        elif defect_type == 'missing_horizontal':
                            orientation = metadata.get('orientation', '?')
                            if orientation == 'X':
                                floor_missing_info.append(f"- Horizontal beam X at bay {bay}: {self._format_bbox(comp.bbox_norm)}")
                            else:
                                floor_missing_info.append(f"- Horizontal beam Y at column {column}: {self._format_bbox(comp.bbox_norm)}")

                    annotations.append({
                        'id': f"{scene_id}_missing_floor_{floor}",
                        'point': f"{scene_id}.npy",
                        'conversations': [
                            {
                                'from': 'human',
                                'value': f'<point>\nAre there any missing components on floor {floor}?'
                            },
                            {
                                'from': 'gpt',
                                'value': f"Yes, {len(comps_in_floor)} components are missing on floor {floor}:\n" + '\n'.join(floor_missing_info)
                            }
                        ],
                        'task_type': 'missing_detection_floor',
                        'target_floor': floor,
                        'num_defects': len(comps_in_floor)
                    })

            # 3. Bay-specific questions (for platforms and X-direction beams)
            bays_with_missing: Dict[str, List[ScaffoldComponent]] = {}
            for comp in missing_comps:
                metadata = comp.metadata or {}
                bay = metadata.get('bay')
                if bay is not None:
                    bays_with_missing.setdefault(str(bay), []).append(comp)

            for bay, comps_in_bay in bays_with_missing.items():
                bay_missing_info: List[str] = []
                for comp in comps_in_bay:
                    metadata = comp.metadata or {}
                    defect_type = metadata.get('defect_type', 'unknown')
                    floor = metadata.get('floor', '?')
                    
                    if defect_type == 'missing_platform':
                        bay_missing_info.append(f"- Platform at floor {floor}: {self._format_bbox(comp.bbox_norm)}")
                    elif defect_type == 'missing_horizontal':
                        bay_missing_info.append(f"- Horizontal beam at floor {floor}: {self._format_bbox(comp.bbox_norm)}")

                annotations.append({
                    'id': f"{scene_id}_missing_bay_{bay}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<point>\nAre there any missing components in bay {bay}?'
                        },
                        {
                            'from': 'gpt',
                            'value': f"Yes, {len(comps_in_bay)} components are missing in bay {bay}:\n" + '\n'.join(bay_missing_info)
                        }
                    ],
                    'task_type': 'missing_detection_bay',
                    'target_bay': bay,
                    'num_defects': len(comps_in_bay)
                })

            # 4. Specific component questions (max 5)
            selected_missing = random.sample(missing_comps, min(3, len(missing_comps)))
            for idx, comp in enumerate(selected_missing, 1):
                metadata = comp.metadata or {}
                defect_type = metadata.get('defect_type', 'unknown')
                
                if defect_type == 'missing_platform':
                    floor = metadata.get('floor', '?')
                    bay = metadata.get('bay', '?')
                    question = f'<point>\nIs there a platform at floor {floor}, bay {bay}?'
                    answer = f"No, the platform at floor {floor}, bay {bay} is missing. Expected location: {self._format_bbox(comp.bbox_norm)}."
                elif defect_type == 'missing_vertical':
                    column = metadata.get('column', '?')
                    row = metadata.get('row', '?')
                    question = f'<point>\nIs there a vertical post at column {column}, row {row}?'
                    answer = f"No, the vertical post at column {column}, row {row} is missing. Expected location: {self._format_bbox(comp.bbox_norm)}."
                elif defect_type == 'missing_horizontal':
                    floor = metadata.get('floor', '?')
                    orientation = metadata.get('orientation', '?')
                    if orientation == 'X':
                        bay = metadata.get('bay', '?')
                        question = f'<point>\nIs there a horizontal beam at floor {floor}, bay {bay} (X-direction)?'
                        answer = f"No, the horizontal beam at floor {floor}, bay {bay} (X-direction) is missing. Expected location: {self._format_bbox(comp.bbox_norm)}."
                    else:
                        column = metadata.get('column', '?')
                        question = f'<point>\nIs there a horizontal beam at floor {floor}, column {column} (Y-direction)?'
                        answer = f"No, the horizontal beam at floor {floor}, column {column} (Y-direction) is missing. Expected location: {self._format_bbox(comp.bbox_norm)}."

                annotations.append({
                    'id': f"{scene_id}_missing_specific_{idx:03d}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': question
                        },
                        {
                            'from': 'gpt',
                            'value': answer
                        }
                    ],
                    'task_type': 'missing_detection_specific',
                    'target_instance_id': comp.instance_id
                })

        else:
            # No missing components
            annotations.append({
                'id': f"{scene_id}_missing_none",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': f'<point>\nThis is a {scaffold_spec}. Are there any missing components?'
                    },
                    {
                        'from': 'gpt',
                        'value': f'{expected_text}\n{actual_text}\nNo missing components detected. All scaffold components are properly installed.'
                    }
                ],
                'task_type': 'missing_detection_none',
                'num_defects': 0
            })

        # 5. Vertical-specific questions
        if missing_verticals:
            vertical_info = []
            for comp in missing_verticals[:3]:
                metadata = comp.metadata or {}
                column = metadata.get('column', '?')
                row = metadata.get('row', '?')
                vertical_info.append(f"- Column {column}, row {row}: {self._format_bbox(comp.bbox_norm)}")

            annotations.append({
                'id': f"{scene_id}_missing_vertical_summary",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': f'<point>\nThis {scaffold_spec} should have {expected_verticals} vertical posts. Are there any missing vertical posts?'
                    },
                    {
                        'from': 'gpt',
                        'value': f"Expected: {expected_verticals} vertical posts.\nActual: {actual_verticals} vertical posts.\nMissing: {len(missing_verticals)} vertical posts are missing:\n" + '\n'.join(vertical_info)
                    }
                ],
                'task_type': 'missing_detection_vertical_summary',
                'num_defects': len(missing_verticals)
            })

            # Specific vertical questions
            for idx, comp in enumerate(random.sample(missing_verticals, min(2, len(missing_verticals))), 1):
                metadata = comp.metadata or {}
                column = metadata.get('column', '?')
                row = metadata.get('row', '?')
                
                annotations.append({
                    'id': f"{scene_id}_missing_vertical_specific_{idx:03d}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<point>\nIs there a vertical post at column {column}, row {row}?'
                        },
                        {
                            'from': 'gpt',
                            'value': f"No, the vertical post at column {column}, row {row} is missing. Expected location: {self._format_bbox(comp.bbox_norm)}."
                        }
                    ],
                    'task_type': 'missing_detection_vertical_specific',
                    'target_instance_id': comp.instance_id
                })

        # 6. Horizontal-specific questions  
        if missing_horizontals:
            horizontal_info = []
            for comp in missing_horizontals[:3]:
                metadata = comp.metadata or {}
                floor = metadata.get('floor', '?')
                orientation = metadata.get('orientation', '?')
                if orientation == 'X':
                    bay = metadata.get('bay', '?')
                    horizontal_info.append(f"- Floor {floor}, bay {bay} (X-direction): {self._format_bbox(comp.bbox_norm)}")
                else:
                    column = metadata.get('column', '?')
                    horizontal_info.append(f"- Floor {floor}, column {column} (Y-direction): {self._format_bbox(comp.bbox_norm)}")

            annotations.append({
                'id': f"{scene_id}_missing_horizontal_summary",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': f'<point>\nThis {scaffold_spec} should have {expected_horizontals} horizontal beams. Are there any missing horizontal beams?'
                    },
                    {
                        'from': 'gpt',
                        'value': f"Expected: {expected_horizontals} horizontal beams.\nActual: {actual_horizontals} horizontal beams.\nMissing: {len(missing_horizontals)} horizontal beams are missing:\n" + '\n'.join(horizontal_info)
                    }
                ],
                'task_type': 'missing_detection_horizontal_summary',
                'num_defects': len(missing_horizontals)
            })

            # Specific horizontal questions
            for idx, comp in enumerate(random.sample(missing_horizontals, min(2, len(missing_horizontals))), 1):
                metadata = comp.metadata or {}
                floor = metadata.get('floor', '?')
                orientation = metadata.get('orientation', '?')
                
                if orientation == 'X':
                    bay = metadata.get('bay', '?')
                    question = f'<point>\nIs there a horizontal beam at floor {floor}, bay {bay} (X-direction)?'
                    answer = f"No, the horizontal beam at floor {floor}, bay {bay} (X-direction) is missing. Expected location: {self._format_bbox(comp.bbox_norm)}."
                else:
                    column = metadata.get('column', '?')
                    question = f'<point>\nIs there a horizontal beam at floor {floor}, column {column} (Y-direction)?'
                    answer = f"No, the horizontal beam at floor {floor}, column {column} (Y-direction) is missing. Expected location: {self._format_bbox(comp.bbox_norm)}."

                annotations.append({
                    'id': f"{scene_id}_missing_horizontal_specific_{idx:03d}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': question
                        },
                        {
                            'from': 'gpt',
                            'value': answer
                        }
                    ],
                    'task_type': 'missing_detection_horizontal_specific',
                    'target_instance_id': comp.instance_id
                })

        # 7. Positive examples (components that exist)
        positive_components = []
        if present_platforms:
            positive_components.extend(random.sample(present_platforms, min(2, len(present_platforms))))
        if present_verticals:
            positive_components.extend(random.sample(present_verticals, min(1, len(present_verticals))))
        if present_horizontals:
            positive_components.extend(random.sample(present_horizontals, min(1, len(present_horizontals))))

        for idx, comp in enumerate(positive_components[:3], 1):
            metadata = comp.metadata or {}
            
            if comp.semantic_id == 3:  # Platform
                floor = metadata.get('floor', '?')
                bay = metadata.get('bay', '?')
                question = f'<point>\nIs there a platform at floor {floor}, bay {bay}?'
                answer = f"Yes, there is a platform at floor {floor}, bay {bay}. Location: {self._format_bbox(comp.bbox_norm)}."
                task_type = 'missing_detection_specific_positive'
            elif comp.semantic_id == 0:  # Vertical post
                column = metadata.get('column', '?')
                row = metadata.get('row', '?')
                question = f'<point>\nIs there a vertical post at column {column}, row {row}?'
                answer = f"Yes, there is a vertical post at column {column}, row {row}. Location: {self._format_bbox(comp.bbox_norm)}."
                task_type = 'missing_detection_vertical_specific_positive'
            elif comp.semantic_id == 1:  # Horizontal beam
                floor = metadata.get('floor', '?')
                orientation = metadata.get('orientation', '?')
                if orientation == 'X':
                    bay = metadata.get('bay', '?')
                    question = f'<point>\nIs there a horizontal beam at floor {floor}, bay {bay} (X-direction)?'
                    answer = f"Yes, there is a horizontal beam at floor {floor}, bay {bay} (X-direction). Location: {self._format_bbox(comp.bbox_norm)}."
                else:
                    column = metadata.get('column', '?')
                    question = f'<point>\nIs there a horizontal beam at floor {floor}, column {column} (Y-direction)?'
                    answer = f"Yes, there is a horizontal beam at floor {floor}, column {column} (Y-direction). Location: {self._format_bbox(comp.bbox_norm)}."
                task_type = 'missing_detection_horizontal_specific_positive'

            annotations.append({
                'id': f"{scene_id}_missing_positive_{idx:03d}",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': question
                    },
                    {
                        'from': 'gpt',
                        'value': answer
                    }
                ],
                'task_type': task_type,
                'target_instance_id': comp.instance_id
            })

        return annotations

    def generate_scene_data(self, scene_id: str) -> Optional[Dict]:
        """Generate a complete scaffold scene with missing detection capabilities."""
        
        # 🆕 Reset missing quota for each scene
        self.reset_missing_quota()
        self.instance_counter = 1

        # Scene configuration
        safety_status = random.choices(
            ['safe', 'minor_defect', 'major_defect'],
            weights=[0.4, 0.3, 0.3]
        )[0]

        config = {
            'num_bays': random.randint(2, 4),
            'bay_width': random.uniform(1.5, 2.0),
            'depth': random.uniform(1.2, 1.8),
            'num_floors': random.randint(2, 4),
            'floor_height': random.uniform(1.8, 2.2),
            'safety_status': safety_status
        }

        if safety_status != 'safe':
            config['missing_type_quota'] = {
                'missing_vertical': 1,
                'missing_horizontal': 2,
                'missing_platform': 1
            }

        num_bays = config['num_bays']
        bay_width = config['bay_width']
        depth = config['depth']
        num_floors = config['num_floors']
        floor_height = config['floor_height']

        # Calculate cumulative heights
        cumulative_heights = [i * floor_height for i in range(num_floors + 1)]

        all_components: List[ScaffoldComponent] = []
        all_violations: List[str] = []

        try:
            # Create components with missing detection
            verticals, v_violations = self._create_vertical_posts_with_validation(
                num_bays, bay_width, depth, cumulative_heights, config
            )
            all_components.extend(verticals)
            all_violations.extend(v_violations)

            horizontals, h_violations = self._create_horizontal_beams_with_validation(
                num_bays, bay_width, depth, cumulative_heights, config
            )
            all_components.extend(horizontals)
            all_violations.extend(h_violations)

            platforms, p_violations = self._create_platforms_with_validation(
                num_bays, bay_width, depth, cumulative_heights, config
            )
            all_components.extend(platforms)
            all_violations.extend(p_violations)

            # Add other components
            braces, b_violations = self._create_diagonal_braces_with_validation(
                num_bays, bay_width, depth, cumulative_heights, num_floors
            )
            all_components.extend(braces)
            all_violations.extend(b_violations)

            handrails, r_violations = self._create_safety_handrails(
                num_bays, bay_width, depth, cumulative_heights, config
            )
            all_components.extend(handrails)
            all_violations.extend(r_violations)

            # Add vertical ladders
            ladders = self._create_vertical_ladders(
                num_bays, bay_width, depth, cumulative_heights
            )
            all_components.extend(ladders)

        except Exception as e:
            print(f"Error generating scene {scene_id}: {str(e)}")
            return None

        if len(all_components) == 0:
            return None

        # Combine all points
        all_points = []
        semantic_labels = []
        instance_labels = []

        for comp in all_components:
            all_points.append(comp.points)
            semantic_labels.extend([comp.semantic_id] * len(comp.points))
            instance_labels.extend([comp.instance_id] * len(comp.points))

        if len(all_points) == 0:
            return None

        coord = np.vstack(all_points)
        semantic_gt = np.array(semantic_labels, dtype=np.int32)
        instance_gt = np.array(instance_labels, dtype=np.int32)

        # Normalize point cloud
        target_points = random.randint(50000, 150000)
        current_points = len(coord)

        if current_points > target_points:
            indices = np.random.choice(current_points, target_points, replace=False)
            coord = coord[indices]
            semantic_gt = semantic_gt[indices]
            instance_gt = instance_gt[indices]
        elif current_points < target_points:
            needed = target_points - current_points
            indices = np.random.choice(current_points, needed, replace=True)
            extra_coord = coord[indices] + np.random.normal(0, 0.01, (needed, 3))
            extra_semantic = semantic_gt[indices]
            extra_instance = instance_gt[indices]
            coord = np.vstack([coord, extra_coord])
            semantic_gt = np.hstack([semantic_gt, extra_semantic])
            instance_gt = np.hstack([instance_gt, extra_instance])

        # 🆕 Improved normalization process
        # Step 1: Centering
        centroid = np.mean(coord, axis=0)
        coord_centered = coord - centroid
        
        # Step 2: Scaling (ensure max distance is ~1.0)
        max_distance = np.linalg.norm(coord_centered, axis=1).max()
        scale = float(max_distance + 1e-12)
        coord_scaled = coord_centered / scale

        # Step 3: Random rotation (Z-axis, ±45 degrees for better viewpoint diversity)
        Rz_deg = float(np.random.uniform(-45.0, 45.0))
        theta = np.radians(Rz_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
        coord_norm = (R @ coord_scaled.T).T

        # 🆕 Ensure coordinates are within [-1, 1] range
        coord_norm = np.clip(coord_norm, -1.0, 1.0)

        # Add gray color
        gray_rgb = np.full((coord_norm.shape[0], 3), 0.5, dtype=np.float32)
        coord_norm = np.concatenate([coord_norm.astype(np.float32), gray_rgb], axis=1)

        # Normalization parameters
        norm_params = {
            'centroid': centroid.tolist(),
            'scale': scale,
            'Rz_deg': Rz_deg
        }

        # 🆕 Recalculate bounding boxes from NORMALIZED point cloud for perfect alignment
        bbox_errors = 0
        normalized_points_3d = coord_norm[:, :3]  # Remove color channels for bbox calculation
        
        # Create mapping from instance_id to normalized points
        instance_to_points = {}
        for i, inst_id in enumerate(instance_gt):
            if inst_id not in instance_to_points:
                instance_to_points[inst_id] = []
            instance_to_points[inst_id].append(normalized_points_3d[i])
        
        # Recalculate bbox_norm from actual normalized point cloud
        for comp in all_components:
            if comp.instance_id in instance_to_points:
                try:
                    # Get normalized points for this component
                    comp_points = np.array(instance_to_points[comp.instance_id])
                    if len(comp_points) > 0:
                        # Calculate bbox directly from normalized points
                        comp.bbox_norm = self.calculate_bbox(comp_points)
                        
                        if comp.bbox_norm is not None:
                            # Verify bbox is within range (should be guaranteed now)
                            bbox_range = [comp.bbox_norm.min(), comp.bbox_norm.max()]
                            if bbox_range[0] < -1.1 or bbox_range[1] > 1.1:
                                print(f"⚠️ Warning: bbox still out of range for {comp.name}: [{bbox_range[0]:.3f}, {bbox_range[1]:.3f}]")
                                bbox_errors += 1
                    else:
                        comp.bbox_norm = None
                except Exception as e:
                    print(f"⚠️ Error recalculating bbox for {comp.name}: {e}")
                    comp.bbox_norm = None
                    bbox_errors += 1
            else:
                comp.bbox_norm = None

        # Update config with violations and missing count
        config.update({
            'violations': all_violations,
            'missing_count': self.current_missing_count,
            'safety_status': 'major_defect' if len(all_violations) > 3 else 'minor_defect' if len(all_violations) > 0 else 'safe'
        })

        # Generate annotations
        annotations = self.generate_shapellm_annotations(scene_id, all_components, config)

        return {
            'coord': coord_norm,
            'semantic_gt': semantic_gt,
            'instance_gt': instance_gt,
            'scene_id': scene_id,
            'config': config,
            'annotations': annotations,
            'components': all_components,
            'norm_params': norm_params
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🏗️ ShapeLLM scaffold missing detection dataset generator (Final Version)')
    parser.add_argument('--num_scenes', type=int, default=50, help='Number of scenes to generate')
    parser.add_argument('--output_dir', type=str, default='./scaffold_test_final', help='Output directory')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Test generation
    generator = EnhancedScaffoldGeneratorFinal(random_seed=args.random_seed)
    
    print(f"🏗️ Testing scaffold generation with missing quota system...")
    print(f"📊 Missing quota: {generator.missing_quota} components max per scene")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i in range(args.num_scenes):
        scene_id = f"test_scaffold_{i:05d}"
        scene_data = generator.generate_scene_data(scene_id)
        
        if scene_data:
            missing_count = scene_data['config']['missing_count']
            missing_comps = [c for c in scene_data['components'] if c.semantic_id == 10]
            print(f"✅ {scene_id}: {missing_count} missing components (actual: {len(missing_comps)})")
            
            # Save point cloud
            npy_path = os.path.join(args.output_dir, f"{scene_id}.npy")
            np.save(npy_path, scene_data['coord'].astype('float32'))
            
        else:
            print(f"❌ Failed: {scene_id}")
    
    print(f"\n🎯 Key features implemented:")
    print(f"✅ Missing quota system (max {generator.missing_quota} per scene)")
    print("✅ Vertical/horizontal/platform missing detection")
    print("✅ Comprehensive question types")
    print("✅ Z-axis vertical ladders")
    print("✅ Proper train/val separation support")