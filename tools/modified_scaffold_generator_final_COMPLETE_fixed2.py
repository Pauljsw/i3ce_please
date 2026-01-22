"""
🏗️ 고도화된 비계 합성 데이터 생성 도구 (ShapeLLM용) - 개선 버전
- 포인트 생성 연속성 개선 (띄엄띄엄 → 연속적)
- V/H/P 누락 비율 균형화 (부재 개수 차이 보정)

개선 사항:
1. generate_pipe_points: i % 5 조건 제거 → 모든 포인트에서 radial variation 생성
2. 누락 비율: 부재 유형별 개수 차이를 보정하는 weighted missing rate 적용
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
    """비계 부품 정의"""
    name: str
    semantic_id: int
    instance_id: int
    points: np.ndarray
    bbox: Optional[np.ndarray] = None
    bbox_norm: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class KoreanScaffoldRegulations:
    """한국 산업안전보건기준 (2025년)"""
    MAX_COLUMN_SPACING_LEDGER = 1.85
    MAX_COLUMN_SPACING_PURLIN = 1.5
    MIN_PLATFORM_WIDTH = 0.40
    MAX_PLATFORM_GAP = 0.03
    TOP_RAIL_HEIGHT_MIN = 0.90
    TOP_RAIL_HEIGHT_MAX = 1.20
    MID_RAIL_REQUIRED = True
    TOE_BOARD_MIN_HEIGHT = 0.10
    MAX_BRACE_VERTICAL_SPAN = 5
    BRACE_ANGLE_MIN = 40
    BRACE_ANGLE_MAX = 60
    MAX_WALL_TIE_SPACING = 5.0

    @classmethod
    def check_column_spacing(cls, spacing_x: float, spacing_y: float) -> List[str]:
        violations: List[str] = []
        if spacing_x > cls.MAX_COLUMN_SPACING_LEDGER:
            violations.append(f"Column spacing exceeded in ledger direction: {spacing_x:.2f} m > {cls.MAX_COLUMN_SPACING_LEDGER} m")
        if spacing_y > cls.MAX_COLUMN_SPACING_PURLIN:
            violations.append(f"Column spacing exceeded in purlin direction: {spacing_y:.2f} m > {cls.MAX_COLUMN_SPACING_PURLIN} m")
        return violations

    @classmethod
    def check_platform_width(cls, width: float) -> List[str]:
        if width < cls.MIN_PLATFORM_WIDTH:
            return [f"Platform width insufficient: {width:.2f} m < {cls.MIN_PLATFORM_WIDTH} m"]
        return []


class ScaffoldSpecs:
    """비계 부품 규격"""
    VERTICAL_LENGTHS = {'V-38': 3.8, 'V-19': 1.9, 'V-09': 0.95, 'V-04': 0.475}
    HORIZONTAL_SPECS = {
        'H-18': {'length': 1.768, 'spacing': 1.817},
        'H-15': {'length': 1.463, 'spacing': 1.512},
        'H-12': {'length': 1.158, 'spacing': 1.207},
        'H-09': {'length': 0.853, 'spacing': 0.902},
        'H-06': {'length': 0.549, 'spacing': 0.598},
        'H-03': {'length': 0.244, 'spacing': 0.293}
    }
    DIAGONAL_SPECS = {
        'B-1918': {'length': 2.629, 'height': 1.9, 'width': 1.829},
        'B-1915': {'length': 2.428, 'height': 1.9, 'width': 1.524},
        'B-1912': {'length': 2.251, 'height': 1.9, 'width': 1.219}
    }
    PLATFORM_SIZES = [(0.4, 0.598), (0.4, 0.902), (0.4, 1.817)]
    BASE_SUPPORT = {'base_size': (0.14, 0.14), 'pipe_diameter': 0.034, 'height': 0.15}
    PIPE_DIAMETERS = {'vertical': 0.0486, 'horizontal': 0.0427, 'diagonal': 0.034, 'handrail': 0.034}


class ImprovedScaffoldGenerator:
    """개선된 비계 생성기 - 포인트 연속성 및 누락 비율 균형화"""

    def __init__(self, random_seed: Optional[int] = 42) -> None:
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        self.class_names: List[str] = [
            "vertical post", "horizontal beam", "diagonal brace", "platform",
            "base support", "connection", "stair", "ladder", "safety rail",
            "damaged component", "missing part"
        ]
        self.class_names_en: List[str] = list(self.class_names)
        
        # Missing quota system
        self.missing_quota = 4
        self.current_missing_count = 0
        self.current_missing_by_type: Dict[str, int] = defaultdict(int)
        self.instance_counter = 1

    def reset_missing_quota(self) -> None:
        self.current_missing_count = 0
        self.current_missing_by_type = defaultdict(int)

    def can_add_missing_component(self) -> bool:
        return self.current_missing_count < self.missing_quota

    def add_missing_component(self, defect_type: Optional[str] = None) -> bool:
        if self.can_add_missing_component():
            self.current_missing_count += 1
            if defect_type:
                self.current_missing_by_type[defect_type] += 1
            return True
        return False

    def generate_pipe_points(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                             diameter: float, num_points: int = 50) -> np.ndarray:
        """
        🔧 개선: 연속적인 파이프 포인트 생성
        - 기존: i % 5 == 0 조건으로 5개마다 하나만 생성 → 띄엄띄엄
        - 개선: 모든 포인트에서 radial variation 생성 → 연속적
        """
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return np.array([])
        
        direction_norm = direction / length
        points = []
        
        # 🔧 개선: perpendicular vectors를 미리 계산 (효율성)
        if abs(direction_norm[2]) < 0.9:
            perp1 = np.cross(direction_norm, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(direction_norm, np.array([1, 0, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction_norm, perp1)
        
        # 🔧 개선: 모든 위치에서 radial points 생성 (i % 5 조건 제거)
        num_radial = 6  # 원형 단면당 포인트 수 (8 → 6으로 줄여서 총 포인트 수 조절)
        
        for i in range(num_points):
            t = i / max(num_points - 1, 1)
            center = start_pos + t * direction
            
            # 중심점 추가
            points.append(center)
            
            # 🔧 개선: 모든 위치에서 radial variation 생성
            for j in range(num_radial):
                angle = j * 2 * np.pi / num_radial
                # 약간의 랜덤성 추가하여 더 자연스럽게
                radius = diameter / 2 * (0.9 + 0.2 * random.random())
                offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                points.append(center + offset)
        
        return np.array(points)

    def generate_platform_points(self, center: np.ndarray, width: float, 
                                  length: float, num_points: int = 100) -> np.ndarray:
        """발판 포인트 생성 - 더 균일한 분포"""
        points = []
        
        # 🔧 개선: 격자 기반 + 노이즈로 더 균일한 분포
        grid_x = int(np.sqrt(num_points * width / length))
        grid_y = int(np.sqrt(num_points * length / width))
        
        for i in range(grid_x):
            for j in range(grid_y):
                x_offset = -width/2 + (i + 0.5) * width / grid_x
                y_offset = -length/2 + (j + 0.5) * length / grid_y
                # 약간의 랜덤 노이즈
                x_offset += random.uniform(-width/(3*grid_x), width/(3*grid_x))
                y_offset += random.uniform(-length/(3*grid_y), length/(3*grid_y))
                z_offset = random.uniform(-0.01, 0.01)
                points.append(center + np.array([x_offset, y_offset, z_offset]))
        
        # 부족한 포인트 추가
        while len(points) < num_points:
            x_offset = random.uniform(-width/2, width/2)
            y_offset = random.uniform(-length/2, length/2)
            z_offset = random.uniform(-0.01, 0.01)
            points.append(center + np.array([x_offset, y_offset, z_offset]))
        
        return np.array(points[:num_points])

    def calculate_bbox(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return None
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        corners = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]]
        ])
        return corners

    def normalize_bbox(self, bbox: np.ndarray, centroid: np.ndarray, 
                       scale: float, R: np.ndarray) -> np.ndarray:
        if bbox is None:
            return None
        bbox_centered = bbox - centroid
        bbox_scaled = bbox_centered / scale
        bbox_rotated = (R @ bbox_scaled.T).T
        bbox_clipped = np.clip(bbox_rotated, -1.0, 1.0)
        return bbox_clipped.astype(np.float32)

    def calculate_balanced_missing_rates(self, num_bays: int, num_floors: int, 
                                         safety_status: str) -> Dict[str, float]:
        """
        🔧 핵심 개선: 부재 유형별 개수를 고려한 균형 잡힌 누락률 계산
        
        목표: V, H, P 각 유형에서 기대 누락 개수가 비슷하도록 조정
        """
        # 1. 각 유형별 총 부재 개수 계산
        num_verticals = (num_bays + 1) * 2  # 예: 3-bay → 8개
        num_horizontals = num_floors * (num_bays * 2 + (num_bays + 1))  # 예: 3-bay, 4-floor → 40개
        num_platforms = num_floors * num_bays  # 예: 3-bay, 4-floor → 12개
        
        total_components = num_verticals + num_horizontals + num_platforms
        
        # 2. 기본 누락률 설정 (전체 기준)
        base_total_missing = {
            'safe': 0,
            'minor_defect': 2,  # 전체 2개 누락 목표
            'major_defect': 4   # 전체 4개 누락 목표 (quota와 동일)
        }[safety_status]
        
        if base_total_missing == 0:
            return {'vertical': 0.0, 'horizontal': 0.0, 'platform': 0.0}
        
        # 3. 각 유형별 목표 누락 개수 (균등 분배)
        # V, H, P 각각에서 비슷한 개수가 누락되도록
        target_missing_per_type = base_total_missing / 3
        
        # 4. 역산하여 각 유형별 누락률 계산
        # missing_rate = target_missing / total_count
        # 단, 최소 1개는 누락될 수 있도록 보정
        
        vertical_rate = min(target_missing_per_type / max(num_verticals, 1), 0.5)  # 최대 50%
        horizontal_rate = min(target_missing_per_type / max(num_horizontals, 1), 0.3)  # 최대 30% (개수가 많으므로)
        platform_rate = min(target_missing_per_type / max(num_platforms, 1), 0.4)  # 최대 40%
        
        # 5. 약간의 랜덤성 추가 (±20%)
        vertical_rate *= random.uniform(0.8, 1.2)
        horizontal_rate *= random.uniform(0.8, 1.2)
        platform_rate *= random.uniform(0.8, 1.2)
        
        rates = {
            'vertical': vertical_rate,
            'horizontal': horizontal_rate,
            'platform': platform_rate
        }
        
        print(f"  📊 부재 개수: V={num_verticals}, H={num_horizontals}, P={num_platforms}")
        print(f"  📊 조정된 누락률: V={vertical_rate:.1%}, H={horizontal_rate:.1%}, P={platform_rate:.1%}")
        print(f"  📊 기대 누락 개수: V={num_verticals*vertical_rate:.1f}, H={num_horizontals*horizontal_rate:.1f}, P={num_platforms*platform_rate:.1f}")
        
        return rates

    def _create_vertical_posts_with_validation(
        self, num_bays: int, bay_width: float, depth: float,
        cumulative_heights: List[float], config: Dict
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """수직재 생성 (개선된 누락률 적용)"""
        components: List[ScaffoldComponent] = []
        violations: List[str] = []

        # 🔧 개선: 균형 잡힌 누락률 사용
        missing_rates = config.get('balanced_missing_rates', {})
        missing_rate = missing_rates.get('vertical', 0.0)

        diameter = ScaffoldSpecs.PIPE_DIAMETERS['vertical']
        max_height = max(cumulative_heights)

        all_verticals = []
        for col in range(num_bays + 1):
            for row in range(2):
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

        # 누락 처리
        if missing_rate > 0 and len(all_verticals) > 0:
            num_to_remove = max(1, int(len(all_verticals) * missing_rate))
            num_to_remove = min(num_to_remove, self.missing_quota - self.current_missing_count)

            if num_to_remove > 0:
                random.shuffle(all_verticals)
                for i in range(num_to_remove):
                    comp = all_verticals[i]
                    col = comp.metadata['column']
                    row = comp.metadata['row']
                    x, y = comp.metadata['position']

                    mid_height = max_height / 2
                    marker_points = np.array([[x, y, mid_height] + np.random.normal(0, 0.05, 3) for _ in range(10)])
                    bbox = self.calculate_bbox(marker_points)

                    all_verticals[i] = ScaffoldComponent(
                        name=f"missing_vertical_{col}_{row}_{comp.instance_id}",
                        semantic_id=10,
                        instance_id=comp.instance_id,
                        points=marker_points,
                        bbox=bbox,
                        metadata={'defect_type': 'missing_vertical', 'column': col, 'row': row, 'floor': 'all'}
                    )
                    self.add_missing_component('missing_vertical')
                    violations.append(f"Missing vertical post at column {col}, row {row}")

        components.extend(all_verticals)
        return components, violations

    def _create_horizontal_beams_with_validation(
        self, num_bays: int, bay_width: float, depth: float,
        cumulative_heights: List[float], config: Dict
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """수평재 생성 (개선된 누락률 적용)"""
        components: List[ScaffoldComponent] = []
        violations: List[str] = []

        missing_rates = config.get('balanced_missing_rates', {})
        missing_rate = missing_rates.get('horizontal', 0.0)

        diameter = ScaffoldSpecs.PIPE_DIAMETERS['horizontal']

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
                            metadata={'orientation': 'X', 'floor': floor_idx, 'bay': bay, 'side': side_idx, 'mid_pos': mid_pos}
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
                        metadata={'orientation': 'Y', 'floor': floor_idx, 'column': col, 'side': 0, 'mid_pos': mid_pos}
                    )
                    all_horizontals.append(comp)
                    self.instance_counter += 1

        # 누락 처리
        if missing_rate > 0 and len(all_horizontals) > 0:
            num_to_remove = max(1, int(len(all_horizontals) * missing_rate))
            num_to_remove = min(num_to_remove, self.missing_quota - self.current_missing_count)

            if num_to_remove > 0:
                random.shuffle(all_horizontals)
                for i in range(num_to_remove):
                    comp = all_horizontals[i]
                    floor_idx = comp.metadata['floor']
                    mid_pos = comp.metadata['mid_pos']
                    orientation = comp.metadata['orientation']

                    marker_points = np.array([mid_pos + np.random.normal(0, 0.05, 3) for _ in range(10)])
                    bbox = self.calculate_bbox(marker_points)

                    if orientation == 'X':
                        bay = comp.metadata['bay']
                        name = f"missing_horizontal_X_{bay}_{floor_idx}_{comp.instance_id}"
                        violation_msg = f"Missing horizontal beam X at floor {floor_idx}, bay {bay}"
                    else:
                        col = comp.metadata['column']
                        name = f"missing_horizontal_Y_{col}_{floor_idx}_{comp.instance_id}"
                        violation_msg = f"Missing horizontal beam Y at floor {floor_idx}, column {col}"

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
        self, num_bays: int, bay_width: float, depth: float,
        cumulative_heights: List[float], config: Dict
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """발판 생성 (개선된 누락률 적용)"""
        components: List[ScaffoldComponent] = []
        violations: List[str] = []

        missing_rates = config.get('balanced_missing_rates', {})
        missing_rate = missing_rates.get('platform', 0.0)

        all_platforms = []
        for floor_idx, z in enumerate(cumulative_heights[:-1]):
            for bay in range(num_bays):
                platform_center = np.array([(bay + 0.5) * bay_width, depth / 2, z])
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
                        metadata={'width': platform_width, 'floor': floor_idx, 'bay': bay, 'center': platform_center}
                    )
                    all_platforms.append(component)
                    self.instance_counter += 1

        # 누락 처리
        if missing_rate > 0 and len(all_platforms) > 0:
            num_to_remove = max(1, int(len(all_platforms) * missing_rate))
            num_to_remove = min(num_to_remove, self.missing_quota - self.current_missing_count)

            if num_to_remove > 0:
                random.shuffle(all_platforms)
                for i in range(num_to_remove):
                    comp = all_platforms[i]
                    floor_idx = comp.metadata['floor']
                    bay = comp.metadata['bay']
                    center = comp.metadata['center']

                    marker_points = np.array([[center[0] + random.uniform(-0.1, 0.1), 
                                               center[1] + random.uniform(-0.1, 0.1), 
                                               center[2]] for _ in range(10)])
                    bbox = self.calculate_bbox(marker_points)

                    all_platforms[i] = ScaffoldComponent(
                        name=f"missing_platform_{floor_idx}_{bay}_{comp.instance_id}",
                        semantic_id=10,
                        instance_id=comp.instance_id,
                        points=marker_points,
                        bbox=bbox,
                        metadata={'defect_type': 'missing_platform', 'floor': floor_idx, 'bay': bay}
                    )
                    self.add_missing_component('missing_platform')
                    violations.append(f"Missing platform at floor {floor_idx}, bay {bay}")

        components.extend(all_platforms)
        return components, violations

    def _create_diagonal_braces_with_validation(
        self, num_bays: int, bay_width: float, depth: float,
        cumulative_heights: List[float], num_floors: int
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """대각재 생성"""
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

        if floors_without_braces:
            consecutive = 1
            for i in range(1, len(floors_without_braces)):
                if floors_without_braces[i] == floors_without_braces[i-1] + 1:
                    consecutive += 1
                    if consecutive >= KoreanScaffoldRegulations.MAX_BRACE_VERTICAL_SPAN:
                        violations.append(f"Braces not installed: {consecutive} consecutive floors")
                        break
                else:
                    consecutive = 1

        return components, violations

    def _create_safety_handrails(
        self, num_bays: int, bay_width: float, depth: float,
        cumulative_heights: List[float], config: Dict
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """안전난간 생성"""
        components: List[ScaffoldComponent] = []
        violations: List[str] = []
        safety_status = config.get('safety_status', 'safe')
        diameter = ScaffoldSpecs.PIPE_DIAMETERS['handrail']

        for floor_idx in range(1, len(cumulative_heights) - 1):
            z = cumulative_heights[floor_idx]
            top_rail_height = random.uniform(
                KoreanScaffoldRegulations.TOP_RAIL_HEIGHT_MIN,
                KoreanScaffoldRegulations.TOP_RAIL_HEIGHT_MAX
            )
            
            if safety_status in ['minor_defect', 'major_defect'] and random.random() < 0.3:
                violations.append(f"Missing safety handrail on floor {floor_idx}")
                continue
            
            rail_z = z + top_rail_height
            
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

    def _create_vertical_ladders(
        self, num_bays: int, bay_width: float, depth: float,
        cumulative_heights: List[float]
    ) -> List[ScaffoldComponent]:
        """수직 사다리 생성"""
        components: List[ScaffoldComponent] = []
        
        for floor_idx in range(0, len(cumulative_heights)-1, 2):
            if floor_idx + 1 >= len(cumulative_heights):
                break
                
            ladder_bay = random.randint(0, max(0, num_bays-1))
            ladder_x = (ladder_bay + 0.5) * bay_width
            ladder_y = depth * 0.1
            z_bottom = cumulative_heights[floor_idx]
            z_top = cumulative_heights[floor_idx + 1]
            diameter = ScaffoldSpecs.PIPE_DIAMETERS['handrail']
            
            for rail_offset in [-0.3, 0.3]:
                start_pos = np.array([ladder_x + rail_offset, ladder_y, z_bottom])
                end_pos = np.array([ladder_x + rail_offset, ladder_y, z_top])
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

    def _format_bbox(self, bbox: Optional[np.ndarray]) -> str:
        if bbox is None or len(bbox) != 8:
            return "N/A"
        corners_list = [[float(f"{corner[0]:.3f}"), float(f"{corner[1]:.3f}"), float(f"{corner[2]:.3f}")]
                        for corner in bbox]
        return str(corners_list)

    def generate_shapellm_annotations(self, scene_id: str, components: List[ScaffoldComponent], 
                                       config: Dict) -> List[Dict]:
        """ShapeLLM 어노테이션 생성 (간소화 버전)"""
        annotations: List[Dict] = []
        
        num_bays = config.get('num_bays', 3)
        num_floors = config.get('num_floors', 4)
        scaffold_spec = f"{num_bays}-bay, 2-row, {num_floors}-floor scaffold"

        missing_comps = [c for c in components if c.semantic_id == 10]
        present_verticals = [c for c in components if c.semantic_id == 0]
        present_horizontals = [c for c in components if c.semantic_id == 1]
        present_platforms = [c for c in components if c.semantic_id == 3]

        missing_verticals = [c for c in missing_comps if (c.metadata or {}).get('defect_type') == 'missing_vertical']
        missing_horizontals = [c for c in missing_comps if (c.metadata or {}).get('defect_type') == 'missing_horizontal']
        missing_platforms = [c for c in missing_comps if (c.metadata or {}).get('defect_type') == 'missing_platform']

        expected_verticals = len(present_verticals) + len(missing_verticals)
        expected_horizontals = len(present_horizontals) + len(missing_horizontals)
        expected_platforms = len(present_platforms) + len(missing_platforms)

        expected_text = f"Expected: {expected_verticals} vertical posts, {expected_horizontals} horizontal beams, {expected_platforms} platforms."
        actual_text = f"Actual: {len(present_verticals)} vertical posts, {len(present_horizontals)} horizontal beams, {len(present_platforms)} platforms."

        if missing_comps:
            missing_info = []
            for comp in missing_comps[:5]:
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
                    missing_info.append(f"- Horizontal beam {orientation} at floor {floor}: {self._format_bbox(comp.bbox_norm)}")

            annotations.append({
                'id': f"{scene_id}_missing_summary",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {'from': 'human', 'value': f'<point>\nThis is a {scaffold_spec}. Are there any missing components?'},
                    {'from': 'gpt', 'value': f"{expected_text}\n{actual_text}\nMissing: {len(missing_comps)} components:\n" + '\n'.join(missing_info)}
                ],
                'task_type': 'missing_detection_summary',
                'num_defects': len(missing_comps)
            })
        else:
            annotations.append({
                'id': f"{scene_id}_missing_none",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {'from': 'human', 'value': f'<point>\nThis is a {scaffold_spec}. Are there any missing components?'},
                    {'from': 'gpt', 'value': f'{expected_text}\n{actual_text}\nNo missing components detected.'}
                ],
                'task_type': 'missing_detection_none',
                'num_defects': 0
            })

        return annotations

    def generate_scene_data(self, scene_id: str) -> Optional[Dict]:
        """씬 데이터 생성 (개선된 버전)"""
        self.reset_missing_quota()
        self.instance_counter = 1

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

        # 🔧 핵심: 균형 잡힌 누락률 계산
        if safety_status != 'safe':
            config['balanced_missing_rates'] = self.calculate_balanced_missing_rates(
                config['num_bays'], config['num_floors'], safety_status
            )
        else:
            config['balanced_missing_rates'] = {'vertical': 0.0, 'horizontal': 0.0, 'platform': 0.0}

        num_bays = config['num_bays']
        bay_width = config['bay_width']
        depth = config['depth']
        num_floors = config['num_floors']
        floor_height = config['floor_height']
        cumulative_heights = [i * floor_height for i in range(num_floors + 1)]

        all_components: List[ScaffoldComponent] = []
        all_violations: List[str] = []

        try:
            verticals, v_violations = self._create_vertical_posts_with_validation(
                num_bays, bay_width, depth, cumulative_heights, config)
            all_components.extend(verticals)
            all_violations.extend(v_violations)

            horizontals, h_violations = self._create_horizontal_beams_with_validation(
                num_bays, bay_width, depth, cumulative_heights, config)
            all_components.extend(horizontals)
            all_violations.extend(h_violations)

            platforms, p_violations = self._create_platforms_with_validation(
                num_bays, bay_width, depth, cumulative_heights, config)
            all_components.extend(platforms)
            all_violations.extend(p_violations)

            braces, b_violations = self._create_diagonal_braces_with_validation(
                num_bays, bay_width, depth, cumulative_heights, num_floors)
            all_components.extend(braces)
            all_violations.extend(b_violations)

            handrails, r_violations = self._create_safety_handrails(
                num_bays, bay_width, depth, cumulative_heights, config)
            all_components.extend(handrails)
            all_violations.extend(r_violations)

            ladders = self._create_vertical_ladders(num_bays, bay_width, depth, cumulative_heights)
            all_components.extend(ladders)

        except Exception as e:
            print(f"Error generating scene {scene_id}: {str(e)}")
            return None

        if len(all_components) == 0:
            return None

        # 포인트 결합
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

        # 포인트 수 조정
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
            coord = np.vstack([coord, extra_coord])
            semantic_gt = np.hstack([semantic_gt, semantic_gt[indices]])
            instance_gt = np.hstack([instance_gt, instance_gt[indices]])

        # 정규화
        centroid = np.mean(coord, axis=0)
        coord_centered = coord - centroid
        max_distance = np.linalg.norm(coord_centered, axis=1).max()
        scale = float(max_distance + 1e-12)
        coord_scaled = coord_centered / scale

        Rz_deg = float(np.random.uniform(-45.0, 45.0))
        theta = np.radians(Rz_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
        coord_norm = (R @ coord_scaled.T).T
        coord_norm = np.clip(coord_norm, -1.0, 1.0)

        gray_rgb = np.full((coord_norm.shape[0], 3), 0.5, dtype=np.float32)
        coord_norm = np.concatenate([coord_norm.astype(np.float32), gray_rgb], axis=1)

        norm_params = {'centroid': centroid.tolist(), 'scale': scale, 'Rz_deg': Rz_deg}

        # bbox 재계산
        normalized_points_3d = coord_norm[:, :3]
        instance_to_points = {}
        for i, inst_id in enumerate(instance_gt):
            if inst_id not in instance_to_points:
                instance_to_points[inst_id] = []
            instance_to_points[inst_id].append(normalized_points_3d[i])
        
        for comp in all_components:
            if comp.instance_id in instance_to_points:
                try:
                    comp_points = np.array(instance_to_points[comp.instance_id])
                    if len(comp_points) > 0:
                        comp.bbox_norm = self.calculate_bbox(comp_points)
                except Exception:
                    comp.bbox_norm = None
            else:
                comp.bbox_norm = None

        config.update({
            'violations': all_violations,
            'missing_count': self.current_missing_count,
            'missing_by_type': dict(self.current_missing_by_type)
        })

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


def analyze_missing_distribution(results: List[Dict]) -> None:
    """누락 분포 분석"""
    print("\n" + "="*60)
    print("📊 누락 분포 분석 결과")
    print("="*60)
    
    total_v, total_h, total_p = 0, 0, 0
    missing_v, missing_h, missing_p = 0, 0, 0
    
    for result in results:
        config = result['config']
        missing_by_type = config.get('missing_by_type', {})
        
        # 부재 개수 계산
        num_bays = config['num_bays']
        num_floors = config['num_floors']
        
        total_v += (num_bays + 1) * 2
        total_h += num_floors * (num_bays * 2 + (num_bays + 1))
        total_p += num_floors * num_bays
        
        missing_v += missing_by_type.get('missing_vertical', 0)
        missing_h += missing_by_type.get('missing_horizontal', 0)
        missing_p += missing_by_type.get('missing_platform', 0)
    
    print(f"\n총 부재 개수:")
    print(f"  - Vertical:   {total_v:4d} (누락: {missing_v:3d}, 누락률: {missing_v/total_v*100:.1f}%)")
    print(f"  - Horizontal: {total_h:4d} (누락: {missing_h:3d}, 누락률: {missing_h/total_h*100:.1f}%)")
    print(f"  - Platform:   {total_p:4d} (누락: {missing_p:3d}, 누락률: {missing_p/total_p*100:.1f}%)")
    
    print(f"\n누락 개수 비교 (목표: 균등):")
    print(f"  - V:H:P = {missing_v}:{missing_h}:{missing_p}")
    
    total_missing = missing_v + missing_h + missing_p
    if total_missing > 0:
        print(f"  - 비율: V={missing_v/total_missing*100:.0f}%, H={missing_h/total_missing*100:.0f}%, P={missing_p/total_missing*100:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='개선된 비계 합성 데이터 생성기')
    parser.add_argument('--num_scenes', type=int, default=20, help='생성할 씬 수')
    parser.add_argument('--output_dir', type=str, default='./scaffold_improved_test', help='출력 디렉토리')
    parser.add_argument('--random_seed', type=int, default=42, help='랜덤 시드')
    
    args = parser.parse_args()
    
    generator = ImprovedScaffoldGenerator(random_seed=args.random_seed)
    
    print(f"🏗️ 개선된 비계 생성기 테스트")
    print(f"  - 포인트 연속성 개선 (i % 5 조건 제거)")
    print(f"  - 누락률 균형화 (부재 개수 보정)")
    print(f"  - 생성 씬 수: {args.num_scenes}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = []
    for i in range(args.num_scenes):
        scene_id = f"improved_scaffold_{i:05d}"
        print(f"\n{'='*50}")
        print(f"Scene {i+1}/{args.num_scenes}: {scene_id}")
        
        scene_data = generator.generate_scene_data(scene_id)
        
        if scene_data:
            config = scene_data['config']
            missing_by_type = config.get('missing_by_type', {})
            
            print(f"  ✅ 생성 완료")
            print(f"     Status: {config['safety_status']}")
            print(f"     Missing: V={missing_by_type.get('missing_vertical', 0)}, "
                  f"H={missing_by_type.get('missing_horizontal', 0)}, "
                  f"P={missing_by_type.get('missing_platform', 0)}")
            
            # 저장
            npy_path = os.path.join(args.output_dir, f"{scene_id}.npy")
            np.save(npy_path, scene_data['coord'].astype('float32'))
            
            results.append(scene_data)
        else:
            print(f"  ❌ 생성 실패")
    
    # 누락 분포 분석
    analyze_missing_distribution(results)
    
    print(f"\n🎯 개선 사항 요약:")
    print("  1. 포인트 연속성: generate_pipe_points에서 i % 5 조건 제거")
    print("  2. 누락률 균형: calculate_balanced_missing_rates로 부재 개수 보정")