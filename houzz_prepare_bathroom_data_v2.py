"""
Script to convert bathroom JSON data to ATISS-compatible format.

This script processes JSON files containing bathroom layouts and converts them
to the format expected by ATISS for training.

Input: JSON files with architecture (walls, floors, doors, windows) and furniture (toilet, vanity, tub, shower)
Output: NPZ files compatible with ATISS training pipeline
"""

import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from collections import defaultdict


class BathroomDataPreprocessor:
    """Preprocessor for bathroom scene data to ATISS format."""
    
    # Define object categories
    ARCHITECTURE_TYPES = {'wall', 'floor', 'door', 'window'}
    FURNITURE_TYPES = {'toilet', 'vanity', 'sink', 'tub', 'shower', 'bathtub'}
    
    def __init__(self):
        # Create category mappings
        self.category_to_idx = {
            'toilet': 0,
            'vanity': 1,
            'sink': 1,  # Map sink to vanity
            'shower': 2,
            'tub': 3,
            'bathtub': 3,  # Map bathtub to tub
        }
        self.idx_to_category = {v: k for k, v in self.category_to_idx.items() if k in {'toilet', 'vanity', 'shower', 'tub'}}
        self.num_categories = len(self.idx_to_category)
        
    def load_json(self, json_path: str) -> Dict:
        """Load JSON file."""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def extract_floor_plan_info(self, items: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """Extract floor plan boundaries and architecture info."""
        architecture_items = [item for item in items if item['object_type'] in self.ARCHITECTURE_TYPES]
        
        # Get room boundaries from floor items
        floor_items = [item for item in items if item['object_type'] == 'floor']
        
        if not floor_items:
            # Fallback: use walls to determine boundaries
            wall_items = [item for item in items if item['object_type'] == 'wall']
            if wall_items:
                x_coords = []
                z_coords = []
                for wall in wall_items:
                    x_coords.extend([
                        wall['center_x'] - wall['size_x']/2,
                        wall['center_x'] + wall['size_x']/2
                    ])
                    z_coords.extend([
                        wall['center_z'] - wall['size_z']/2,
                        wall['center_z'] + wall['size_z']/2
                    ])
                min_x, max_x = min(x_coords), max(x_coords)
                min_z, max_z = min(z_coords), max(z_coords)
            else:
                return None, None
        else:
            # Use ALL floor polygons to determine room size (handles L-shaped rooms)
            min_x = min(f['center_x'] - f['size_x']/2 for f in floor_items)
            max_x = max(f['center_x'] + f['size_x']/2 for f in floor_items)
            min_z = min(f['center_z'] - f['size_z']/2 for f in floor_items)
            max_z = max(f['center_z'] + f['size_z']/2 for f in floor_items)
        
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
        
        return architecture_items, room_dims
    
    def normalize_coordinates(self, x: float, z: float, room_dims: Dict) -> Tuple[float, float]:
        """Normalize coordinates to [-1, 1] range relative to room center."""
        # Translate to room center
        x_norm = (x - room_dims['center_x']) / (room_dims['width'] / 2)
        z_norm = (z - room_dims['center_z']) / (room_dims['depth'] / 2)
        return x_norm, z_norm
    
    def normalize_size(self, size_x: float, size_z: float, room_dims: Dict) -> Tuple[float, float]:
        """Normalize size relative to room dimensions."""
        size_x_norm = size_x / room_dims['width']
        size_z_norm = size_z / room_dims['depth']
        return size_x_norm, size_z_norm
    
    def normalize_angle(self, yaw: float) -> float:
        """Normalize angle to [-pi, pi] range."""
        yaw_rad = np.deg2rad(yaw)
        # Normalize to [-pi, pi]
        while yaw_rad > np.pi:
            yaw_rad -= 2 * np.pi
        while yaw_rad < -np.pi:
            yaw_rad += 2 * np.pi
        return yaw_rad
    
    def extract_furniture(self, items: List[Dict], room_dims: Dict) -> List[Dict]:
        """Extract and normalize furniture items."""
        furniture_items = []
        
        for item in items:
            obj_type = item.get('object_type', item.get('diffusion_type', ''))
            
            # Check if this is furniture
            if obj_type in self.FURNITURE_TYPES:
                # Get category index
                category_idx = self.category_to_idx.get(obj_type, -1)
                if category_idx == -1:
                    continue
                
                # Normalize position
                x_norm, z_norm = self.normalize_coordinates(
                    item['center_x'], 
                    item['center_z'], 
                    room_dims
                )
                
                # Normalize size
                size_x_norm, size_z_norm = self.normalize_size(
                    item['size_x'],
                    item['size_z'],
                    room_dims
                )
                
                # Normalize angle
                angle_norm = self.normalize_angle(item['yaw'])
                
                furniture_items.append({
                    'category': category_idx,
                    'x': x_norm,
                    'z': z_norm,
                    'size_x': size_x_norm,
                    'size_z': size_z_norm,
                    'angle': angle_norm,
                    'uuid': item['uuid'],
                    'model_name': item['model_name']
                })
        
        return furniture_items
    
    def create_floor_plan_mask(self, architecture_items: List[Dict], room_dims: Dict, 
                                resolution: int = 128) -> np.ndarray:
        """
        Create a binary floor plan mask showing the room bounding box.
        
        For ATISS training, we just need to show where the room is (rectangular boundary).
        The model will learn from training data where furniture actually gets placed within this space.
        
        This avoids issues with overlapping floor polygons in the JSON data.
        """
        from PIL import Image, ImageDraw
        
        # Create a simple rectangular mask showing the room bounds
        # Start with black (outside room)
        img = Image.new('L', (resolution, resolution), color=0)
        draw = ImageDraw.Draw(img)
        
        # Calculate padding (10% on each side)
        padding = int(resolution * 0.1)
        
        # Draw the room as a white rectangle (walkable area)
        draw.rectangle(
            [padding, padding, resolution - padding, resolution - padding],
            fill=255
        )
        
        # Convert to numpy and binarize
        mask = np.array(img)
        mask = (mask > 127).astype(np.uint8)
        
        return mask
    
    def process_scene(self, json_path: str, output_dir: str, scene_id: str) -> bool:
        """Process a single scene and save in ATISS format."""
        try:
            # Load JSON
            data = self.load_json(json_path)
            items = data.get('items', [])
            
            if not items:
                print(f"Warning: No items found in {json_path}")
                return False
            
            # Extract architecture and room dimensions
            architecture_items, room_dims = self.extract_floor_plan_info(items)
            
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
            
            # Class labels (one-hot encoded)
            class_labels = np.zeros((num_objects, self.num_categories), dtype=np.float32)
            for i, item in enumerate(furniture_items):
                class_labels[i, item['category']] = 1.0
            
            # Translations (x, z positions)
            translations = np.array([[item['x'], item['z']] for item in furniture_items], dtype=np.float32)
            
            # Sizes (width, depth)
            sizes = np.array([[item['size_x'], item['size_z']] for item in furniture_items], dtype=np.float32)
            
            # Angles (yaw in radians)
            angles = np.array([[np.cos(item['angle']), np.sin(item['angle'])] 
                               for item in furniture_items], dtype=np.float32)
            
            # Create floor plan mask
            room_mask = self.create_floor_plan_mask(architecture_items, room_dims)
            
            # Save to NPZ file
            scene_output_dir = os.path.join(output_dir, scene_id)
            os.makedirs(scene_output_dir, exist_ok=True)
            
            np.savez_compressed(
                os.path.join(scene_output_dir, 'boxes.npz'),
                class_labels=class_labels,
                translations=translations,
                sizes=sizes,
                angles=angles,
                room_layout=room_mask,
                objfeats_32=np.zeros((num_objects, 32), dtype=np.float32),  # Placeholder for object features
            )
            
            # Save room mask as image for visualization
            from PIL import Image
            mask_img = Image.fromarray((room_mask * 255).astype(np.uint8))
            mask_img.save(os.path.join(scene_output_dir, 'room_mask.png'))
            
            # Save metadata
            metadata = {
                'scene_id': scene_id,
                'num_objects': num_objects,
                'room_dims': room_dims,
                'furniture_items': [{'uuid': item['uuid'], 'model_name': item['model_name'], 
                                    'category': self.idx_to_category[item['category']]} 
                                   for item in furniture_items]
            }
            
            with open(os.path.join(scene_output_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error processing {json_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_dataset(self, input_dir: str, output_dir: str):
        """Process all JSON files in the input directory."""
        input_path = Path(input_dir)
        json_files = list(input_path.glob('*.json'))
        
        print(f"Found {len(json_files)} JSON files")
        
        stats = {'success': 0, 'failed': 0}
        
        for json_file in json_files:
            scene_id = json_file.stem
            print(f"Processing {scene_id}...")
            
            if self.process_scene(str(json_file), output_dir, scene_id):
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {stats['success']}")
        print(f"Failed: {stats['failed']}")
        
        # Save dataset statistics
        dataset_stats = {
            'total_scenes': len(json_files),
            'successful': stats['success'],
            'failed': stats['failed'],
            'categories': self.idx_to_category,
            'num_categories': self.num_categories
        }
        
        with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
            json.dump(dataset_stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Prepare bathroom data for ATISS training')
    parser.add_argument('input_dir', type=str, help='Directory containing JSON files')
    parser.add_argument('output_dir', type=str, help='Output directory for processed data')
    parser.add_argument('--resolution', type=int, default=64, 
                       help='Resolution for room mask (default: 64)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process dataset
    preprocessor = BathroomDataPreprocessor()
    preprocessor.process_dataset(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
