"""
Scaffold Point Cloud Generator Module

This module generates synthetic scaffold point clouds with configurable
missing components for training and evaluation.

Key Features:
- Geometry generation following Korean safety standards
- Missing component simulation with quota system
- Proper bounding box calculation and normalization
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

from .config import ScaffoldConfig, ScaffoldGeometryConfig, MissingConfig


@dataclass
class ScaffoldComponent:
    """Represents a single scaffold component with its properties."""

    name: str
    semantic_id: int
    instance_id: int
    points: np.ndarray  # [N, 3] XYZ coordinates
    bbox: Optional[np.ndarray] = None  # [8, 3] world coordinates
    bbox_norm: Optional[np.ndarray] = None  # [8, 3] normalized coordinates
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'semantic_id': self.semantic_id,
            'instance_id': self.instance_id,
            'num_points': len(self.points),
            'bbox_norm': self.bbox_norm.tolist() if self.bbox_norm is not None else None,
            'metadata': self.metadata
        }


class ScaffoldGenerator:
    """Generator for synthetic scaffold point clouds."""

    # Semantic IDs for component types
    SEMANTIC_IDS = {
        'vertical_post': 0,
        'horizontal_beam': 1,
        'diagonal_brace': 2,
        'platform': 3,
        'base_support': 4,
        'connection': 5,
        'stair': 6,
        'ladder': 7,
        'safety_rail': 8,
        'damaged': 9,
        'missing': 10
    }

    def __init__(self, config: ScaffoldConfig):
        """Initialize generator with configuration."""
        self.config = config
        self.geo_config = config.geometry
        self.missing_config = config.missing

        # Set random seeds for reproducibility
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        # Instance counter for unique IDs
        self.instance_counter = 0

        # Missing component tracking
        self.current_missing_count = 0
        self.current_missing_by_type: Dict[str, int] = defaultdict(int)

    def reset_scene(self) -> None:
        """Reset counters for new scene generation."""
        self.instance_counter = 0
        self.current_missing_count = 0
        self.current_missing_by_type = defaultdict(int)

    def _get_next_instance_id(self) -> int:
        """Get next unique instance ID."""
        self.instance_counter += 1
        return self.instance_counter

    def _can_add_missing(self, defect_type: str) -> bool:
        """Check if we can add another missing component."""
        if self.current_missing_count >= self.missing_config.max_missing_per_scene:
            return False

        type_limits = {
            'missing_vertical': self.missing_config.max_missing_vertical,
            'missing_horizontal': self.missing_config.max_missing_horizontal,
            'missing_platform': self.missing_config.max_missing_platform
        }

        current_type_count = self.current_missing_by_type[defect_type]
        max_type_count = type_limits.get(defect_type, 1)

        return current_type_count < max_type_count

    def _add_missing(self, defect_type: str) -> bool:
        """Try to add a missing component. Returns True if successful."""
        if self._can_add_missing(defect_type):
            self.current_missing_count += 1
            self.current_missing_by_type[defect_type] += 1
            return True
        return False

    def generate_pipe_points(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        diameter: float
    ) -> np.ndarray:
        """Generate dense points along a cylindrical pipe."""
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)

        if length < 1e-6:
            return np.array([]).reshape(0, 3)

        direction_norm = direction / length

        # Adaptive sampling: ~100 points per meter
        num_length_samples = max(int(length * 100), 10)
        num_circumference = 12

        # Create perpendicular vectors
        if abs(direction_norm[2]) < 0.9:
            perp1 = np.cross(direction_norm, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(direction_norm, np.array([1, 0, 0]))
        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-12)
        perp2 = np.cross(direction_norm, perp1)

        points = []
        radius = diameter / 2

        for i in range(num_length_samples):
            t = i / max(num_length_samples - 1, 1)
            center = start_pos + t * direction

            for j in range(num_circumference):
                angle = j * 2 * np.pi / num_circumference
                r_var = radius * (1.0 + np.random.uniform(-0.05, 0.05))
                offset = r_var * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                noise = np.random.uniform(-0.005, 0.005) * direction_norm
                points.append(center + offset + noise)

        return np.array(points) if points else np.array([]).reshape(0, 3)

    def generate_platform_points(
        self,
        center: np.ndarray,
        width: float,
        length: float
    ) -> np.ndarray:
        """Generate dense points for a rectangular platform."""
        # Area-based sampling: ~500 points per m^2
        area = width * length
        num_points = max(int(area * 500), 50)
        thickness = 0.03

        points = []

        # Grid-based sampling for top surface
        grid_x = max(int(np.sqrt(num_points * width / length)), 5)
        grid_y = max(int(np.sqrt(num_points * length / width)), 5)

        for i in range(grid_x):
            for j in range(grid_y):
                x = -width/2 + width * (i + np.random.uniform(0.2, 0.8)) / grid_x
                y = -length/2 + length * (j + np.random.uniform(0.2, 0.8)) / grid_y

                # Top surface
                z_top = np.random.uniform(-0.002, 0.002)
                points.append(center + np.array([x, y, z_top]))

                # Bottom surface (30% probability)
                if np.random.random() < 0.3:
                    z_bottom = -thickness + np.random.uniform(-0.002, 0.002)
                    points.append(center + np.array([x, y, z_bottom]))

        # Edge points
        num_edge_points = int(2 * (width + length) * 50)
        for _ in range(num_edge_points):
            edge = np.random.randint(4)
            if edge == 0:
                x, y = -width/2, np.random.uniform(-length/2, length/2)
            elif edge == 1:
                x, y = width/2, np.random.uniform(-length/2, length/2)
            elif edge == 2:
                x, y = np.random.uniform(-width/2, width/2), -length/2
            else:
                x, y = np.random.uniform(-width/2, width/2), length/2

            x += np.random.uniform(-0.005, 0.005)
            y += np.random.uniform(-0.005, 0.005)
            z = np.random.uniform(-thickness, 0)
            points.append(center + np.array([x, y, z]))

        return np.array(points) if points else np.array([]).reshape(0, 3)

    def calculate_bbox(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Calculate 8-corner bounding box from points."""
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

    def _create_missing_marker(
        self,
        position: np.ndarray,
        defect_type: str,
        metadata: Dict
    ) -> np.ndarray:
        """Create marker points for a missing component."""
        marker_points = []
        for _ in range(10):
            noise = np.random.normal(0, 0.05, 3)
            marker_points.append(position + noise)
        return np.array(marker_points)

    def _create_vertical_posts(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        max_height: float,
        safety_status: str
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """Create vertical posts with potential missing components."""
        components = []
        violations = []

        diameter = self.geo_config.vertical_diameter

        # Create all posts first
        all_posts = []
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
                        name=f"vertical_post_{self._get_next_instance_id()}",
                        semantic_id=self.SEMANTIC_IDS['vertical_post'],
                        instance_id=self.instance_counter,
                        points=points,
                        bbox=bbox,
                        metadata={'column': col, 'row': row, 'position': (x, y)}
                    )
                    all_posts.append(comp)

        # Apply missing based on safety status
        if safety_status != 'safe':
            missing_rate = (
                self.missing_config.minor_defect_rate
                if safety_status == 'minor_defect'
                else self.missing_config.major_defect_rate
            )

            num_candidates = max(1, int(len(all_posts) * missing_rate))
            random.shuffle(all_posts)

            for i in range(min(num_candidates, len(all_posts))):
                if not self._can_add_missing('missing_vertical'):
                    break

                comp = all_posts[i]
                col = comp.metadata['column']
                row = comp.metadata['row']
                x, y = comp.metadata['position']

                # Convert to missing marker
                mid_height = max_height / 2
                marker_pos = np.array([x, y, mid_height])
                marker_points = self._create_missing_marker(
                    marker_pos, 'missing_vertical', comp.metadata
                )
                marker_bbox = self.calculate_bbox(marker_points)

                all_posts[i] = ScaffoldComponent(
                    name=f"missing_vertical_{col}_{row}",
                    semantic_id=self.SEMANTIC_IDS['missing'],
                    instance_id=comp.instance_id,
                    points=marker_points,
                    bbox=marker_bbox,
                    metadata={
                        'defect_type': 'missing_vertical',
                        'column': col,
                        'row': row,
                        'floor': 'all'
                    }
                )
                self._add_missing('missing_vertical')
                violations.append(f"Missing vertical post at column {col}, row {row}")

        components.extend(all_posts)
        return components, violations

    def _create_horizontal_beams(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float],
        safety_status: str
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """Create horizontal beams with potential missing components."""
        components = []
        violations = []

        diameter = self.geo_config.horizontal_diameter

        # Create all beams first
        all_beams = []
        for floor_idx, z in enumerate(cumulative_heights[:-1]):
            # X-direction beams
            for bay in range(num_bays):
                for side_idx, y in enumerate([0, depth]):
                    start_pos = np.array([bay * bay_width, y, z])
                    end_pos = np.array([(bay + 1) * bay_width, y, z])
                    mid_pos = (start_pos + end_pos) / 2.0

                    points = self.generate_pipe_points(start_pos, end_pos, diameter)
                    if len(points) > 0:
                        bbox = self.calculate_bbox(points)
                        comp = ScaffoldComponent(
                            name=f"horizontal_beam_X_{self._get_next_instance_id()}",
                            semantic_id=self.SEMANTIC_IDS['horizontal_beam'],
                            instance_id=self.instance_counter,
                            points=points,
                            bbox=bbox,
                            metadata={
                                'orientation': 'X',
                                'floor': floor_idx,
                                'bay': bay,
                                'side': side_idx,
                                'mid_pos': mid_pos.tolist()
                            }
                        )
                        all_beams.append(comp)

            # Y-direction beams
            for col in range(num_bays + 1):
                start_pos = np.array([col * bay_width, 0, z])
                end_pos = np.array([col * bay_width, depth, z])
                mid_pos = (start_pos + end_pos) / 2.0

                points = self.generate_pipe_points(start_pos, end_pos, diameter)
                if len(points) > 0:
                    bbox = self.calculate_bbox(points)
                    comp = ScaffoldComponent(
                        name=f"horizontal_beam_Y_{self._get_next_instance_id()}",
                        semantic_id=self.SEMANTIC_IDS['horizontal_beam'],
                        instance_id=self.instance_counter,
                        points=points,
                        bbox=bbox,
                        metadata={
                            'orientation': 'Y',
                            'floor': floor_idx,
                            'column': col,
                            'side': 0,
                            'mid_pos': mid_pos.tolist()
                        }
                    )
                    all_beams.append(comp)

        # Apply missing based on safety status
        if safety_status != 'safe':
            missing_rate = (
                self.missing_config.minor_defect_rate
                if safety_status == 'minor_defect'
                else self.missing_config.major_defect_rate
            )

            num_candidates = max(1, int(len(all_beams) * missing_rate))
            random.shuffle(all_beams)

            for i in range(min(num_candidates, len(all_beams))):
                if not self._can_add_missing('missing_horizontal'):
                    break

                comp = all_beams[i]
                metadata = comp.metadata

                # Convert to missing marker
                mid_pos = np.array(metadata['mid_pos'])
                marker_points = self._create_missing_marker(
                    mid_pos, 'missing_horizontal', metadata
                )
                marker_bbox = self.calculate_bbox(marker_points)

                all_beams[i] = ScaffoldComponent(
                    name=f"missing_horizontal_{metadata['floor']}_{metadata.get('bay', metadata.get('column', 0))}",
                    semantic_id=self.SEMANTIC_IDS['missing'],
                    instance_id=comp.instance_id,
                    points=marker_points,
                    bbox=marker_bbox,
                    metadata={
                        'defect_type': 'missing_horizontal',
                        'orientation': metadata['orientation'],
                        'floor': metadata['floor'],
                        'bay': metadata.get('bay'),
                        'column': metadata.get('column'),
                        'side': metadata.get('side')
                    }
                )
                self._add_missing('missing_horizontal')
                violations.append(
                    f"Missing horizontal beam at floor {metadata['floor']}, "
                    f"{'bay ' + str(metadata.get('bay')) if metadata['orientation'] == 'X' else 'column ' + str(metadata.get('column'))}"
                )

        components.extend(all_beams)
        return components, violations

    def _create_platforms(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float],
        safety_status: str
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """Create platforms with potential missing components."""
        components = []
        violations = []

        platform_width = 0.4
        platform_length = bay_width

        # Create all platforms first
        all_platforms = []
        for floor_idx, z in enumerate(cumulative_heights[:-1]):
            for bay in range(num_bays):
                center = np.array([
                    bay * bay_width + bay_width / 2,
                    depth / 2,
                    z
                ])

                points = self.generate_platform_points(center, platform_width, platform_length)
                if len(points) > 0:
                    bbox = self.calculate_bbox(points)
                    comp = ScaffoldComponent(
                        name=f"platform_{self._get_next_instance_id()}",
                        semantic_id=self.SEMANTIC_IDS['platform'],
                        instance_id=self.instance_counter,
                        points=points,
                        bbox=bbox,
                        metadata={
                            'floor': floor_idx,
                            'bay': bay,
                            'center': center.tolist()
                        }
                    )
                    all_platforms.append(comp)

        # Apply missing based on safety status
        if safety_status != 'safe':
            missing_rate = (
                self.missing_config.minor_defect_rate
                if safety_status == 'minor_defect'
                else self.missing_config.major_defect_rate
            )

            num_candidates = max(1, int(len(all_platforms) * missing_rate))
            random.shuffle(all_platforms)

            for i in range(min(num_candidates, len(all_platforms))):
                if not self._can_add_missing('missing_platform'):
                    break

                comp = all_platforms[i]
                metadata = comp.metadata

                # Convert to missing marker
                center = np.array(metadata['center'])
                marker_points = self._create_missing_marker(
                    center, 'missing_platform', metadata
                )
                marker_bbox = self.calculate_bbox(marker_points)

                all_platforms[i] = ScaffoldComponent(
                    name=f"missing_platform_{metadata['floor']}_{metadata['bay']}",
                    semantic_id=self.SEMANTIC_IDS['missing'],
                    instance_id=comp.instance_id,
                    points=marker_points,
                    bbox=marker_bbox,
                    metadata={
                        'defect_type': 'missing_platform',
                        'floor': metadata['floor'],
                        'bay': metadata['bay']
                    }
                )
                self._add_missing('missing_platform')
                violations.append(f"Missing platform at floor {metadata['floor']}, bay {metadata['bay']}")

        components.extend(all_platforms)
        return components, violations

    def _create_diagonal_braces(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float]
    ) -> List[ScaffoldComponent]:
        """Create diagonal braces (no missing simulation for now)."""
        components = []
        diameter = self.geo_config.diagonal_diameter

        num_floors = len(cumulative_heights) - 1

        # Add diagonal braces every 2-3 bays
        for bay in range(0, num_bays, random.randint(2, 3)):
            for floor_idx in range(num_floors - 1):
                z_low = cumulative_heights[floor_idx]
                z_high = cumulative_heights[floor_idx + 1]

                # Front diagonal
                start = np.array([bay * bay_width, 0, z_low])
                end = np.array([(bay + 1) * bay_width, 0, z_high])
                points = self.generate_pipe_points(start, end, diameter)

                if len(points) > 0:
                    bbox = self.calculate_bbox(points)
                    components.append(ScaffoldComponent(
                        name=f"diagonal_brace_{self._get_next_instance_id()}",
                        semantic_id=self.SEMANTIC_IDS['diagonal_brace'],
                        instance_id=self.instance_counter,
                        points=points,
                        bbox=bbox,
                        metadata={'floor': floor_idx, 'bay': bay, 'side': 'front'}
                    ))

        return components

    def _create_safety_handrails(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float]
    ) -> List[ScaffoldComponent]:
        """Create safety handrails on top level."""
        components = []
        diameter = 0.034  # handrail diameter
        top_z = cumulative_heights[-1]
        rail_height = 1.0  # 1m height

        # Front and back rails
        for y in [0, depth]:
            start = np.array([0, y, top_z])
            end = np.array([num_bays * bay_width, y, top_z + rail_height])

            points = self.generate_pipe_points(
                np.array([0, y, top_z + rail_height]),
                np.array([num_bays * bay_width, y, top_z + rail_height]),
                diameter
            )

            if len(points) > 0:
                bbox = self.calculate_bbox(points)
                components.append(ScaffoldComponent(
                    name=f"safety_rail_{self._get_next_instance_id()}",
                    semantic_id=self.SEMANTIC_IDS['safety_rail'],
                    instance_id=self.instance_counter,
                    points=points,
                    bbox=bbox,
                    metadata={'side': 'front' if y == 0 else 'back'}
                ))

        return components

    def normalize_point_cloud(
        self,
        points: np.ndarray,
        components: List[ScaffoldComponent]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalize point cloud and update component bounding boxes.

        Returns:
            Tuple of (normalized_points, normalization_params)
        """
        # Step 1: Centering
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid

        # Step 2: Scaling (max distance ~1.0)
        max_distance = np.linalg.norm(points_centered, axis=1).max()
        scale = float(max_distance + 1e-12)
        points_scaled = points_centered / scale

        # Step 3: Random rotation (Z-axis, ±45 degrees)
        Rz_deg = float(np.random.uniform(-45.0, 45.0))
        theta = np.radians(Rz_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        points_norm = (R @ points_scaled.T).T

        # Clip to [-1, 1]
        points_norm = np.clip(points_norm, -1.0, 1.0)

        norm_params = {
            'centroid': centroid.tolist(),
            'scale': scale,
            'Rz_deg': Rz_deg,
            'R': R.tolist()
        }

        # Update component bounding boxes
        for comp in components:
            if comp.bbox is not None:
                bbox_centered = comp.bbox - centroid
                bbox_scaled = bbox_centered / scale
                bbox_rotated = (R @ bbox_scaled.T).T
                comp.bbox_norm = np.clip(bbox_rotated, -1.0, 1.0).astype(np.float32)

        return points_norm.astype(np.float32), norm_params

    def generate_scene(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate a complete scaffold scene.

        Returns:
            Dictionary containing point cloud, components, and metadata
        """
        self.reset_scene()

        # Determine safety status
        safety_probs = [
            self.missing_config.safe_ratio,
            self.missing_config.minor_defect_ratio,
            self.missing_config.major_defect_ratio
        ]
        safety_status = random.choices(
            ['safe', 'minor_defect', 'major_defect'],
            weights=safety_probs
        )[0]

        # Random geometry parameters
        num_bays = random.randint(
            self.geo_config.min_bays,
            self.geo_config.max_bays
        )
        bay_width = random.uniform(*self.geo_config.bay_width_range)
        depth = random.uniform(*self.geo_config.depth_range)
        num_floors = random.randint(
            self.geo_config.min_floors,
            self.geo_config.max_floors
        )
        floor_height = random.uniform(*self.geo_config.floor_height_range)

        cumulative_heights = [i * floor_height for i in range(num_floors + 1)]
        max_height = cumulative_heights[-1]

        # Generate components
        all_components = []
        all_violations = []

        # Vertical posts
        verticals, v_violations = self._create_vertical_posts(
            num_bays, bay_width, depth, max_height, safety_status
        )
        all_components.extend(verticals)
        all_violations.extend(v_violations)

        # Horizontal beams
        horizontals, h_violations = self._create_horizontal_beams(
            num_bays, bay_width, depth, cumulative_heights, safety_status
        )
        all_components.extend(horizontals)
        all_violations.extend(h_violations)

        # Platforms
        platforms, p_violations = self._create_platforms(
            num_bays, bay_width, depth, cumulative_heights, safety_status
        )
        all_components.extend(platforms)
        all_violations.extend(p_violations)

        # Diagonal braces
        braces = self._create_diagonal_braces(
            num_bays, bay_width, depth, cumulative_heights
        )
        all_components.extend(braces)

        # Safety handrails
        rails = self._create_safety_handrails(
            num_bays, bay_width, depth, cumulative_heights
        )
        all_components.extend(rails)

        if len(all_components) == 0:
            return None

        # Combine all points
        all_points = np.vstack([c.points for c in all_components])
        semantic_labels = []
        instance_labels = []

        for comp in all_components:
            semantic_labels.extend([comp.semantic_id] * len(comp.points))
            instance_labels.extend([comp.instance_id] * len(comp.points))

        semantic_labels = np.array(semantic_labels, dtype=np.int32)
        instance_labels = np.array(instance_labels, dtype=np.int32)

        # Sample/upsample to target range
        target_points = random.randint(*self.geo_config.target_points_range)
        current_points = len(all_points)

        if current_points > target_points:
            indices = np.random.choice(current_points, target_points, replace=False)
            all_points = all_points[indices]
            semantic_labels = semantic_labels[indices]
            instance_labels = instance_labels[indices]
        elif current_points < target_points:
            needed = target_points - current_points
            indices = np.random.choice(current_points, needed, replace=True)
            extra_points = all_points[indices] + np.random.normal(0, 0.01, (needed, 3))
            all_points = np.vstack([all_points, extra_points])
            semantic_labels = np.hstack([semantic_labels, semantic_labels[indices]])
            instance_labels = np.hstack([instance_labels, instance_labels[indices]])

        # Normalize
        points_norm, norm_params = self.normalize_point_cloud(all_points, all_components)

        # Add gray color
        gray_rgb = np.full((len(points_norm), 3), 0.5, dtype=np.float32)
        coord = np.concatenate([points_norm, gray_rgb], axis=1)

        # Compute expected counts
        expected_verticals = (num_bays + 1) * 2
        expected_horizontals = num_floors * (num_bays * 2 + (num_bays + 1))
        expected_platforms = num_floors * num_bays

        # Actual counts
        present_verticals = len([c for c in all_components if c.semantic_id == 0])
        present_horizontals = len([c for c in all_components if c.semantic_id == 1])
        present_platforms = len([c for c in all_components if c.semantic_id == 3])

        scene_config = {
            'num_bays': num_bays,
            'bay_width': bay_width,
            'depth': depth,
            'num_floors': num_floors,
            'floor_height': floor_height,
            'safety_status': safety_status,
            'violations': all_violations,
            'missing_count': self.current_missing_count,
            'expected_verticals': expected_verticals,
            'expected_horizontals': expected_horizontals,
            'expected_platforms': expected_platforms,
            'actual_verticals': present_verticals,
            'actual_horizontals': present_horizontals,
            'actual_platforms': present_platforms,
            'total_width': num_bays * bay_width,
            'total_height': max_height
        }

        return {
            'scene_id': scene_id,
            'coord': coord,
            'semantic_gt': semantic_labels,
            'instance_gt': instance_labels,
            'components': all_components,
            'config': scene_config,
            'norm_params': norm_params
        }
