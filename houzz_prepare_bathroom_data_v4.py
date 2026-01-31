#!/usr/bin/env python3
"""
Bathroom Data Preprocessing for ATISS Training
Clean version with simplified logic:
- Room bounds: from ALL items
- Floor mask: only draw floors (white)
- Furniture: only toilet, vanity, shower, tub
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from PIL import Image, ImageDraw


class BathroomDataPreprocessor:
    """Converts bathroom JSON files to ATISS training format."""

    # Clear type definitions
    ARCHITECTURE_TYPES = ['wall', 'door', 'window']
    FURNITURE_TYPES = ['toilet', 'vanity', 'shower', 'tub']
    FLOOR_TYPE = 'floor'

    # Category mapping for furniture only
    CATEGORY_MAP = {
        'toilet': 0,
        'vanity': 1,
        'shower': 2,
        'tub': 3
    }

    def __init__(self, resolution: int = 128):
        """
        Initialize preprocessor.

        Args:
            resolution: Target resolution for room masks (default: 128)
        """
        self.resolution = resolution

    def load_json(self, json_path: str) -> Dict:
        """Load bathroom scene JSON."""
        with open(json_path, 'r') as f:
            return json.load(f)

    def extract_room_dims(self, items: List[Dict]) -> Dict:
        """
        Extract room dimensions from ALL items.

        Args:
            items: All items from JSON

        Returns:
            Room dimension dict
        """
        if not items:
            return None

        x_coords = []
        z_coords = []

        # Use ALL items for room bounds
        for item in items:
            x1 = item['center_x'] - item['size_x'] / 2
            x2 = item['center_x'] + item['size_x'] / 2
            z1 = item['center_z'] - item['size_z'] / 2
            z2 = item['center_z'] + item['size_z'] / 2

            x_coords.extend([x1, x2])
            z_coords.extend([z1, z2])

        min_x = min(x_coords)
        max_x = max(x_coords)
        min_z = min(z_coords)
        max_z = max(z_coords)

        room_dims = {
            'min_x': min_x,
            'max_x': max_x,
            'min_z': min_z,
            'max_z': max_z,
            'width': max_x - min_x,
            'depth': max_z - min_z,
            'center_x': (min_x + max_x) / 2,
            'center_z': (min_z + max_z) / 2
        }

        return room_dims

    def create_floor_plan_mask(self, items: List[Dict], room_dims: Dict,
                                resolution: int = 128) -> np.ndarray:
        """
        Create floor plan mask - ONLY draw floors as white.

        Args:
            items: All items from JSON
            room_dims: Room dimension dict
            resolution: Target resolution

        Returns:
            Binary mask (128, 128) where 1=floor, 0=everything else
        """
        from PIL import Image, ImageDraw

        # Fixed scale: pixels per meter
        PIXELS_PER_METER = 40

        # Calculate canvas size
        canvas_width = int(room_dims['width'] * PIXELS_PER_METER)
        canvas_height = int(room_dims['depth'] * PIXELS_PER_METER)

        # Create canvas - start with BLACK (all zeros)
        canvas = Image.new('L', (canvas_width, canvas_height), color=0)
        draw = ImageDraw.Draw(canvas)

        # Draw ONLY floors as white
        for item in items:
            if item.get('diffusion_type') == self.FLOOR_TYPE:
                # Get center and size
                cx = item['center_x']
                cz = item['center_z']
                w = item['size_x']
                d = item['size_z']
                yaw = item.get('yaw', 0.0)

                # Check if rotated
                if abs(yaw) < 0.001:
                    # Axis-aligned rectangle
                    x1 = cx - w / 2
                    x2 = cx + w / 2
                    z1 = cz - d / 2
                    z2 = cz + d / 2

                    # Convert to canvas pixels
                    px1 = int((x1 - room_dims['min_x']) * PIXELS_PER_METER)
                    px2 = int((x2 - room_dims['min_x']) * PIXELS_PER_METER)
                    py1 = int((z1 - room_dims['min_z']) * PIXELS_PER_METER)
                    py2 = int((z2 - room_dims['min_z']) * PIXELS_PER_METER)

                    # Draw white rectangle
                    draw.rectangle([px1, py1, px2, py2], fill=255)
                else:
                    # Rotated rectangle - calculate 4 corners
                    yaw_rad = np.radians(yaw)
                    cos_yaw = np.cos(yaw_rad)
                    sin_yaw = np.sin(yaw_rad)

                    hw = w / 2
                    hd = d / 2

                    # 4 corners relative to center
                    corners_local = [
                        (-hw, -hd),
                        ( hw, -hd),
                        ( hw,  hd),
                        (-hw,  hd)
                    ]

                    # Rotate and translate
                    corners_world = []
                    for lx, lz in corners_local:
                        rx = lx * cos_yaw - lz * sin_yaw
                        rz = lx * sin_yaw + lz * cos_yaw
                        wx = cx + rx
                        wz = cz + rz
                        corners_world.append((wx, wz))

                    # Convert to canvas pixels
                    corners_pixels = []
                    for wx, wz in corners_world:
                        px = int((wx - room_dims['min_x']) * PIXELS_PER_METER)
                        py = int((wz - room_dims['min_z']) * PIXELS_PER_METER)
                        corners_pixels.append((px, py))

                    # Draw rotated polygon
                    draw.polygon(corners_pixels, fill=255)

        # Resize to fit in target resolution with padding
        target_size = int(resolution * 0.8)
        scale_factor = min(target_size / canvas_width, target_size / canvas_height)

        new_width = int(canvas_width * scale_factor)
        new_height = int(canvas_height * scale_factor)
        canvas_resized = canvas.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create final image with padding to center
        final_img = Image.new('L', (resolution, resolution), color=0)
        offset_x = (resolution - new_width) // 2
        offset_y = (resolution - new_height) // 2
        final_img.paste(canvas_resized, (offset_x, offset_y))

        # Convert to numpy and binarize
        mask = np.array(final_img)
        mask = (mask > 127).astype(np.uint8)

        return mask

    def normalize_coordinates(self, x: float, z: float, room_dims: Dict) -> Tuple[float, float]:
        """Normalize coordinates to [-1, 1] range."""
        x_norm = (x - room_dims['center_x']) / (room_dims['width'] / 2)
        z_norm = (z - room_dims['center_z']) / (room_dims['depth'] / 2)
        return x_norm, z_norm

    def normalize_size(self, size_x: float, size_z: float, room_dims: Dict) -> Tuple[float, float]:
        """Normalize size relative to room dimensions."""
        size_x_norm = size_x / room_dims['width']
        size_z_norm = size_z / room_dims['depth']
        return size_x_norm, size_z_norm

    def extract_furniture(self, items: List[Dict], room_dims: Dict) -> List[Dict]:
        """
        Extract furniture items using diffusion_type only.

        Args:
            items: All items from JSON
            room_dims: Room dimension dict

        Returns:
            List of furniture object dicts with normalized coordinates
        """
        furniture = []

        for item in items:
            diffusion_type = item.get('diffusion_type', '')

            # Only process furniture types
            if diffusion_type not in self.FURNITURE_TYPES:
                continue

            # Normalize coordinates and sizes
            x_norm, z_norm = self.normalize_coordinates(
                item['center_x'],
                item['center_z'],
                room_dims
            )
            size_x_norm, size_z_norm = self.normalize_size(
                item['size_x'],
                item['size_z'],
                room_dims
            )

            # Convert angle to (cos, sin)
            yaw_rad = np.radians(item.get('yaw', 0.0))
            angle_cos = np.cos(yaw_rad)
            angle_sin = np.sin(yaw_rad)

            furniture.append({
                'category': diffusion_type,
                'category_id': self.CATEGORY_MAP[diffusion_type],
                'position': (x_norm, z_norm),
                'size': (size_x_norm, size_z_norm),
                'angle': (angle_cos, angle_sin),
                'uuid': item['uuid'],
                'model_name': item.get('model_name', 'Unknown')
            })

        return furniture

    def process_scene(self, json_path: str, output_dir: str, scene_id: str) -> bool:
        """Process a single scene and save in ATISS format."""
        try:
            # Load JSON
            data = self.load_json(json_path)
            items = data.get('items', [])

            if not items:
                print(f"Warning: No items found in {json_path}")
                return False

            # Extract room dimensions from ALL items
            room_dims = self.extract_room_dims(items)

            if room_dims is None:
                print(f"Warning: Could not determine room dimensions for {json_path}")
                return False

            # Extract furniture
            furniture_items = self.extract_furniture(items, room_dims)

            if not furniture_items:
                print(f"Warning: No furniture items found in {json_path}")
                return False

            # Create arrays for ATISS format
            num_objects = len(furniture_items)
            num_categories = len(self.CATEGORY_MAP)

            # Class labels (one-hot encoded)
            class_labels = np.zeros((num_objects, num_categories), dtype=np.float32)
            for i, item in enumerate(furniture_items):
                class_labels[i, item['category_id']] = 1.0

            # Translations (x, z positions)
            translations = np.array([item['position'] for item in furniture_items], dtype=np.float32)

            # Sizes (width, depth)
            sizes = np.array([item['size'] for item in furniture_items], dtype=np.float32)

            # Angles (cos, sin)
            angles = np.array([item['angle'] for item in furniture_items], dtype=np.float32)

            # Room layout mask (only floors are white)
            room_layout = self.create_floor_plan_mask(items, room_dims, self.resolution)

            # Placeholder for object features
            objfeats_32 = np.zeros((num_objects, 32), dtype=np.float32)

            # Create output directory
            scene_output_dir = Path(output_dir) / scene_id
            scene_output_dir.mkdir(parents=True, exist_ok=True)

            # Save NPZ file
            np.savez_compressed(
                scene_output_dir / 'boxes.npz',
                class_labels=class_labels,
                translations=translations,
                sizes=sizes,
                angles=angles,
                room_layout=room_layout,
                objfeats_32=objfeats_32
            )

            # Save room mask as PNG for visualization
            mask_img = Image.fromarray((room_layout * 255).astype(np.uint8))
            mask_img.save(scene_output_dir / 'room_mask.png')

            # Save metadata
            metadata = {
                'scene_id': scene_id,
                'num_objects': num_objects,
                'num_categories': num_categories,
                'categories': self.CATEGORY_MAP,
                'room_dims': room_dims,
                'furniture_items': [
                    {
                        'uuid': item['uuid'],
                        'model_name': item['model_name'],
                        'category': item['category'],
                        'category_id': item['category_id']
                    }
                    for item in furniture_items
                ]
            }

            with open(scene_output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            return True

        except Exception as e:
            print(f"Error processing {json_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def process_dataset(self, input_dir: str, output_dir: str):
        """Process all JSON files in input directory."""
        input_path = Path(input_dir)
        json_files = list(input_path.glob('*.json'))

        print(f"Found {len(json_files)} JSON files")

        success_count = 0
        fail_count = 0

        for json_file in json_files:
            scene_id = json_file.stem
            print(f"Processing {scene_id}...")

            if self.process_scene(str(json_file), output_dir, scene_id):
                success_count += 1
            else:
                fail_count += 1

        print(f"\nProcessing complete!")
        print(f"Successfully processed: {success_count}")
        print(f"Failed: {fail_count}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess bathroom JSON data for ATISS training')
    parser.add_argument('input_dir', type=str, help='Directory containing JSON files')
    parser.add_argument('output_dir', type=str, help='Output directory for preprocessed data')
    parser.add_argument('--resolution', type=int, default=128, help='Resolution for room masks (default: 128)')

    args = parser.parse_args()

    preprocessor = BathroomDataPreprocessor(resolution=args.resolution)
    preprocessor.process_dataset(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
