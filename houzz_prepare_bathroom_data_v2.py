#!/usr/bin/env python3
"""
Bathroom Data Preprocessing for ATISS Training
Version 5: Proper multi-floor polygon rendering with fixed scale
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from PIL import Image, ImageDraw


class BathroomDataPreprocessor:
    """Converts bathroom JSON files to ATISS training format."""
    
    # Object categories mapping
    CATEGORY_MAP = {
        'toilet': 0,
        'vanity': 1,
        'shower': 2,
        'tub': 3,
        'bathtub': 3  # Alias for tub
    }
    
    FURNITURE_TYPES = ['toilet', 'vanity', 'shower', 'tub', 'bathtub', 'sink', 'model']
    ARCHITECTURE_TYPES = ['wall', 'floor', 'door', 'window']
    
    def __init__(self, resolution: int = 128):
        self.resolution = resolution
    
    def load_json(self, json_path: str) -> Dict:
        """Load bathroom scene JSON."""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def extract_floor_plan_info(self, items: List[Dict]) -> Tuple[List[Dict], Dict]:
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
    
    def create_floor_plan_mask(self, architecture_items: List[Dict], room_dims: Dict, 
                                resolution: int = 128) -> np.ndarray:
        """
        Create floor plan mask with FIXED SCALE and centered positioning.
        
        Strategy:
        1. Use fixed scale (40 pixels/meter)
        2. Create canvas large enough for room at this scale
        3. Draw all floor polygons
        4. Pad/crop to center in resolution x resolution image
        """
        from PIL import Image, ImageDraw
        
        # FIXED SCALE: pixels per meter
        PIXELS_PER_METER = 40
        
        # Calculate canvas size needed for room
        canvas_width = int(room_dims['width'] * PIXELS_PER_METER)
        canvas_height = int(room_dims['depth'] * PIXELS_PER_METER)
        
        # Create canvas at actual scale - black background
        canvas = Image.new('L', (canvas_width, canvas_height), color=0)
        draw = ImageDraw.Draw(canvas)
        
        # Draw each floor polygon
        for item in architecture_items:
            if item['object_type'] in ['floor', 'door']:
                # Get corners in world coordinates
                x1 = item['center_x'] - item['size_x'] / 2
                x2 = item['center_x'] + item['size_x'] / 2
                z1 = item['center_z'] - item['size_z'] / 2
                z2 = item['center_z'] + item['size_z'] / 2
                
                # Convert to canvas pixels
                # Translate to origin, then scale
                px1 = int((x1 - room_dims['min_x']) * PIXELS_PER_METER)
                px2 = int((x2 - room_dims['min_x']) * PIXELS_PER_METER)
                py1 = int((z1 - room_dims['min_z']) * PIXELS_PER_METER)
                py2 = int((z2 - room_dims['min_z']) * PIXELS_PER_METER)
                
                # Draw white rectangle
                draw.rectangle([px1, py1, px2, py2], fill=255)
        
        # Now we need to fit this canvas into resolution x resolution
        # with the room centered
        
        # Calculate scale to fit in target resolution (with 10% padding)
        target_size = int(resolution * 0.8)  # Use 80% of resolution
        scale_factor = min(target_size / canvas_width, target_size / canvas_height)
        
        # Resize canvas
        new_width = int(canvas_width * scale_factor)
        new_height = int(canvas_height * scale_factor)
        canvas_resized = canvas.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create final image with padding to center
        final_img = Image.new('L', (resolution, resolution), color=0)
        
        # Calculate position to center the resized canvas
        offset_x = (resolution - new_width) // 2
        offset_y = (resolution - new_height) // 2
        
        # Paste resized canvas at center
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
        """Extract and normalize furniture items."""
        furniture = []
        
        for item in items:
            # Check if this is furniture
            obj_type = item.get('object_type', '')
            diffusion_type = item.get('diffusion_type', '')
            
            # Map to our categories
            category = None
            if diffusion_type in self.CATEGORY_MAP:
                category = diffusion_type
            elif obj_type == 'sink':
                category = 'vanity'
            elif obj_type == 'model' and 'toilet' in item.get('model_name', '').lower():
                category = 'toilet'
            
            if category and category in self.CATEGORY_MAP:
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
                yaw_rad = np.radians(item['yaw'])
                angle_cos = np.cos(yaw_rad)
                angle_sin = np.sin(yaw_rad)
                
                furniture.append({
                    'category': category,
                    'category_id': self.CATEGORY_MAP[category],
                    'position': (x_norm, z_norm),
                    'size': (size_x_norm, size_z_norm),
                    'angle': (angle_cos, angle_sin),
                    'uuid': item['uuid']
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
            class_labels = np.zeros((num_objects, len(self.CATEGORY_MAP)), dtype=np.float32)
            for i, item in enumerate(furniture_items):
                class_labels[i, item['category_id']] = 1.0
            
            # Translations (x, z positions)
            translations = np.array([item['position'] for item in furniture_items], dtype=np.float32)
            
            # Sizes (width, depth)
            sizes = np.array([item['size'] for item in furniture_items], dtype=np.float32)
            
            # Angles (cos, sin)
            angles = np.array([item['angle'] for item in furniture_items], dtype=np.float32)
            
            # Room layout mask
            room_layout = self.create_floor_plan_mask(architecture_items, room_dims, self.resolution)
            
            # Placeholder for object features (ATISS expects this)
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
                'room_dims': room_dims,
                'furniture_items': [
                    {
                        'uuid': item['uuid'],
                        'model_name': [i for i in items if i['uuid'] == item['uuid']][0].get('model_name', 'Unknown'),
                        'category': item['category']
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
