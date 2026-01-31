"""
Bathroom Dataset for ATISS
Loads preprocessed bathroom NPZ files and provides ATISS-compatible interface
"""

import numpy as np
from pathlib import Path
from collections import Counter

# Import from common module (not base)
from .common import BaseDataset, BaseScene


class BathroomBBox:
    """Bounding box for bathroom furniture, compatible with ATISS."""

    def __init__(self, label, centroid, size, z_angle):
        """
        Args:
            label: str, category name (e.g., 'toilet', 'vanity')
            centroid: [x, y, z] position (y is always 0 for floor plan)
            size: [width, height, depth] normalized sizes
            z_angle: float, rotation angle in radians
        """
        self.label = label
        self._centroid = np.array(centroid, dtype=np.float32)
        self._size = np.array(size, dtype=np.float32)
        self.z_angle = z_angle

    @property
    def centroid(self):
        """Return centroid, optionally offset by scene centroid."""
        def inner(offset=np.array([0, 0, 0])):
            return self._centroid + offset
        return inner

    @property
    def size(self):
        """Return size [width, height, depth]."""
        return self._size

    def one_hot_label(self, all_labels):
        """Return one-hot encoding of label."""
        label_idx = all_labels.index(self.label)
        one_hot = np.zeros(len(all_labels), dtype=np.float32)
        one_hot[label_idx] = 1.0
        return one_hot

    def int_label(self, all_labels):
        """Return integer index of label."""
        return all_labels.index(self.label)


class BathroomScene(BaseScene):
    """Scene for bathroom, compatible with ATISS."""

    def __init__(self, scene_id, bboxes, room_mask, room_dims):
        """
        Args:
            scene_id: str, unique scene identifier
            bboxes: list of BathroomBBox objects
            room_mask: (H, W) binary mask, 1=floor, 0=walls/outside
            room_dims: dict with room dimension info
        """
        super().__init__(
            scene_id=scene_id,
            scene_type="bathroom",
            bboxes=bboxes
        )
        self._room_mask = room_mask
        self.room_dims = room_dims

    @property
    def room_mask(self):
        """Return room mask as (H, W, 1) for ATISS compatibility."""
        # ATISS expects (H, W, C) format
        #return self._room_mask[:, :, np.newaxis]
        #return self._room_mask
        if len(self._room_mask.shape) == 2:
            # (H, W) → (H, W, 1)
            return self._room_mask[:, :, np.newaxis]
        elif len(self._room_mask.shape) == 3:
            # (H, W, C) → keep as-is
            return self._room_mask
        else:
            raise ValueError(f"Unexpected room_mask shape: {self._room_mask.shape}")

    @property
    def centroid(self):
        """Return scene centroid [0, 0, 0] since our data is already normalized."""
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    @property
    def bbox(self):
        """Return scene bounding box (for compatibility)."""
        # Return normalized bounds [-1, 1] for x and z
        return (
            np.array([-1.0, 0.0, -1.0], dtype=np.float32),  # min
            np.array([1.0, 0.0, 1.0], dtype=np.float32)     # max
        )

    @property
    def floor_plan_bbox(self):
        """Return floor plan bounding box."""
        return self.bbox

    def ordered_bboxes_with_class_frequencies(self, class_frequencies):
        """Order bboxes by class frequency (most common first)."""
        # Get frequency for each box
        frequencies = [class_frequencies.get(bbox.label, 0) for bbox in self.bboxes]

        # Get positions for tie-breaking
        centroids = np.array([bbox.centroid() for bbox in self.bboxes])

        # Combine frequencies and positions for sorting
        # Higher frequency first, then by position
        freq_array = np.array([[f] for f in frequencies])
        ordering = np.lexsort(np.hstack([centroids, freq_array]).T)[::-1]

        ordered_bboxes = [self.bboxes[i] for i in ordering]
        return ordered_bboxes


class BathroomDataset(BaseDataset):
    """Dataset for bathroom furniture layout generation."""

    # Furniture categories (4 furniture types + 2 special tokens)
    CATEGORIES = ['toilet', 'vanity', 'shower', 'tub']
    # Full class labels including special tokens (for ATISS compatibility)
    # Index: 0=toilet, 1=vanity, 2=shower, 3=tub, 4=start, 5=end
    CLASS_LABELS_WITH_TOKENS = ['toilet', 'vanity', 'shower', 'tub', 'start', 'end']

    def __init__(self, scenes):
        """
        Args:
            scenes: list of BathroomScene objects
        """
        super().__init__(scenes)
        self._object_types = self.CATEGORIES  # Only furniture types
        self._class_labels = self.CLASS_LABELS_WITH_TOKENS  # Include special tokens

        # Compute class frequencies for ordering
        self._compute_class_frequencies()

    def _compute_class_frequencies(self):
        """Compute frequency of each class across all scenes."""
        all_labels = []
        for scene in self.scenes:
            all_labels.extend([bbox.label for bbox in scene.bboxes])

        counter = Counter(all_labels)
        total = sum(counter.values())

        self._class_frequencies = {
            label: count / total
            for label, count in counter.items()
        }

    @property
    def class_labels(self):
        """Return list of class label strings."""
        return self._class_labels

    @property
    def object_types(self):
        """Return list of object type strings."""
        return self._object_types

    @property
    def class_frequencies(self):
        """Return dict of class frequencies."""
        return self._class_frequencies

    @property
    def bounds(self):
        """Return hard-coded bounds for normalization/denormalization."""
        # Hard-coded bounds based on dataset statistics:
        # - Position: [-25, 25] meters (x and z)
        # - Size: [0, 7] meters (width and depth)
        # - Angle: [-π, π] radians (from arctan2)
        return {
            "translations": np.array([
                [-25.0, 0.0, -25.0],  # min (x, y, z)
                [25.0, 0.0, 25.0]      # max (x, y, z)
            ], dtype=np.float32),
            "sizes": np.array([
                [0.0, 0.0, 0.0],       # min
                [7.0, 7.0, 7.0]        # max (use same bound for all dimensions)
            ], dtype=np.float32),
            "angles": np.array([-np.pi, np.pi], dtype=np.float32)  # [-π, π]
        }

    @staticmethod
    def from_dataset_directory(dataset_dir, scene_ids=None):
        """
        Load dataset from directory containing preprocessed NPZ files.

        Args:
            dataset_dir: str or Path, directory with scene subdirectories
            scene_ids: list of str, specific scene IDs to load (None = all)

        Returns:
            BathroomDataset instance
        """
        dataset_path = Path(dataset_dir)

        # Get all scene directories
        scene_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

        # Filter by scene_ids if provided
        if scene_ids is not None:
            scene_dirs = [d for d in scene_dirs if d.name in scene_ids]

        scenes = []
        print(f"Loading {len(scene_dirs)} scenes from {dataset_dir}")

        for scene_dir in sorted(scene_dirs):
            try:
                scene = BathroomDataset._load_scene(scene_dir)
                if scene is not None:
                    scenes.append(scene)
            except Exception as e:
                print(f"Warning: Failed to load scene {scene_dir.name}: {e}")
                continue

        print(f"Successfully loaded {len(scenes)} scenes")

        if len(scenes) == 0:
            raise ValueError(f"No valid scenes found in {dataset_dir}")

        return BathroomDataset(scenes)

    @staticmethod
    def _load_scene(scene_dir):
        """Load a single scene from directory."""
        scene_id = scene_dir.name
        npz_path = scene_dir / 'boxes.npz'

        if not npz_path.exists():
            print(f"Warning: boxes.npz not found in {scene_dir}")
            return None

        # Load NPZ data
        data = np.load(npz_path)

        class_labels_original = data['class_labels']  # (N, 4) - only 4 furniture classes
        translations = data['translations']  # (N, 2) - [x, z] normalized
        sizes = data['sizes']                # (N, 2) - [width, depth] normalized
        angles = data['angles']              # (N, 2) - [cos, sin]
        room_layout = data['room_layout']    # (128, 128)

        N = len(class_labels_original)

        # IMPORTANT: ATISS expects class_labels with 6 columns:
        # [toilet, vanity, shower, tub, start_token, end_token]
        # Our NPZ only has 4 columns, so we need to add 2 columns
        class_labels = np.zeros((N, 6), dtype=np.float32)
        class_labels[:, :4] = class_labels_original  # Copy first 4 columns
        # Columns 4 and 5 (start and end tokens) remain 0 for regular objects

        # Convert to BBox objects
        bboxes = []
        for i in range(N):
            # Get category from one-hot (only look at first 4 columns)
            category_idx = np.argmax(class_labels[i, :4])
            category = BathroomDataset.CATEGORIES[category_idx]

            # Convert 2D position to 3D (add y=0)
            x, z = translations[i]
            centroid = [x, 0.0, z]

            # Convert 2D size to 3D (add height=0.5 as placeholder)
            width, depth = sizes[i]
            size = [width, 0.5, depth]

            # Convert cos/sin to angle
            cos_angle, sin_angle = angles[i]
            z_angle = np.arctan2(sin_angle, cos_angle)

            bbox = BathroomBBox(
                label=category,
                centroid=centroid,
                size=size,
                z_angle=z_angle
            )
            bboxes.append(bbox)

        # Load room dimensions from metadata if available
        import json
        metadata_path = scene_dir / 'metadata.json'
        room_dims = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                room_dims = metadata.get('room_dims', {})

        scene = BathroomScene(
            scene_id=scene_id,
            bboxes=bboxes,
            room_mask=room_layout,
            room_dims=room_dims
        )

        return scene


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bathroom_dataset.py <dataset_dir>")
        sys.exit(1)

    dataset_dir = sys.argv[1]

    # Load dataset
    dataset = BathroomDataset.from_dataset_directory(dataset_dir)

    print(f"\nDataset info:")
    print(f"  Number of scenes: {len(dataset)}")
    print(f"  Class labels: {dataset.class_labels}")
    print(f"  Class frequencies: {dataset.class_frequencies}")

    # Test loading a sample
    print(f"\nSample scene:")
    scene = dataset[0]
    print(f"  Scene ID: {scene.scene_id}")
    print(f"  Number of objects: {scene.nobjects}")
    print(f"  Object types: {scene.object_types}")
    print(f"  Room mask shape: {scene.room_mask.shape}")

    for i, bbox in enumerate(scene.bboxes):
        print(f"    {i}: {bbox.label} at {bbox.centroid()}, size={bbox.size}, angle={bbox.z_angle:.2f}")
