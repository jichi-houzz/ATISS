"""
Bathroom Scene Preprocessing with 4-channel Architecture Support

Generates:
- Channel 0: Floor (walkable area)
- Channel 1: Walls
- Channel 2: Doors  
- Channel 3: Windows

And extracts furniture objects (toilet, vanity, shower, tub)
"""

import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2


# Configuration
FURNITURE_TYPES = ['toilet', 'vanity', 'shower', 'tub']
ARCHITECTURE_TYPES = ['wall', 'door', 'window']
FLOOR_TYPE = 'floor'
SKIP_TYPES = ['clearance', 'ceiling']

# Mask parameters
MASK_SIZE = 128
PIXELS_PER_METER = 40  # Fixed scale for consistent rendering


def get_room_bounds(items):
    """Calculate room bounds from all items (walls, doors, floors, furniture)."""
    all_points = []
    
    for item in items:
        diff_type = item.get('diffusion_type', '')
        
        # Skip clearance and ceiling
        if diff_type in SKIP_TYPES:
            continue
        
        cx = item.get('center_x', 0)
        cz = item.get('center_z', 0)
        sx = item.get('size_x', 0)
        sz = item.get('size_z', 0)
        
        # Add bbox corners
        all_points.extend([
            [cx - sx/2, cz - sz/2],
            [cx + sx/2, cz + sz/2]
        ])
    
    if not all_points:
        return 0, 0, 5, 5
    
    all_points = np.array(all_points)
    min_x, min_z = all_points.min(axis=0)
    max_x, max_z = all_points.max(axis=0)
    
    return min_x, min_z, max_x, max_z


def world_to_pixel(x, z, min_x, min_z, pixels_per_meter):
    """Convert world coordinates to pixel coordinates."""
    px = int((x - min_x) * pixels_per_meter)
    pz = int((z - min_z) * pixels_per_meter)
    return px, pz


def get_rotated_bbox_corners(center_x, center_z, size_x, size_z, yaw_degrees):
    """
    Get 4 corners of a rotated bounding box.
    
    Args:
        center_x, center_z: Center position
        size_x, size_z: Size (width, depth)
        yaw_degrees: Rotation angle in degrees
        
    Returns:
        corners: (4, 2) array of corner points
    """
    # Half sizes
    hx = size_x / 2
    hz = size_z / 2
    
    # Local corners (before rotation)
    local_corners = np.array([
        [-hx, -hz],
        [ hx, -hz],
        [ hx,  hz],
        [-hx,  hz]
    ])
    
    # Rotation matrix (yaw around y-axis, affects x-z plane)
    yaw_rad = np.radians(yaw_degrees)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw,  cos_yaw]
    ])
    
    # Rotate and translate
    rotated_corners = local_corners @ rotation_matrix.T
    world_corners = rotated_corners + np.array([center_x, center_z])
    
    return world_corners


def draw_architecture_item(mask, item, min_x, min_z, pixels_per_meter, channel, thickness=3):
    """
    Draw an architecture item (wall/door/window) to a specific channel.
    
    Args:
        mask: (H, W, 4) mask array
        item: Item dictionary
        min_x, min_z: Room bounds
        pixels_per_meter: Scale factor
        channel: Which channel to draw to (1=wall, 2=door, 3=window)
        thickness: Line thickness for drawing
    """
    cx = item.get('center_x', 0)
    cz = item.get('center_z', 0)
    sx = item.get('size_x', 0)
    sz = item.get('size_z', 0)
    yaw = item.get('yaw', 0)
    
    # Get rotated corners
    corners = get_rotated_bbox_corners(cx, cz, sx, sz, yaw)
    
    # Convert to pixel coordinates
    pixel_corners = []
    for x, z in corners:
        px, pz = world_to_pixel(x, z, min_x, min_z, pixels_per_meter)
        pixel_corners.append([px, pz])
    
    pixel_corners = np.array(pixel_corners, dtype=np.int32)
    
    # Draw based on type
    diff_type = item.get('diffusion_type', '')
    
    if diff_type == 'wall':
        # Draw wall as outline (hollow rectangle)
        cv2.polylines(mask[:, :, channel], [pixel_corners], 
                     isClosed=True, color=255, thickness=thickness)
    else:
        # Draw door/window as filled rectangle
        cv2.fillPoly(mask[:, :, channel], [pixel_corners], color=255)


def draw_floor_polygon(mask, floor_item, min_x, min_z, pixels_per_meter):
    """Draw floor polygon to channel 0."""
    cx = floor_item.get('center_x', 0)
    cz = floor_item.get('center_z', 0)
    sx = floor_item.get('size_x', 0)
    sz = floor_item.get('size_z', 0)
    yaw = floor_item.get('yaw', 0)
    
    # Get rotated corners
    corners = get_rotated_bbox_corners(cx, cz, sx, sz, yaw)
    
    # Convert to pixel coordinates
    pixel_corners = []
    for x, z in corners:
        px, pz = world_to_pixel(x, z, min_x, min_z, pixels_per_meter)
        pixel_corners.append([px, pz])
    
    pixel_corners = np.array(pixel_corners, dtype=np.int32)
    
    # Draw filled polygon
    cv2.fillPoly(mask[:, :, 0], [pixel_corners], color=255)


def create_4channel_room_layout(items, pixels_per_meter=PIXELS_PER_METER):
    """
    Create 4-channel room layout mask.
    
    Returns:
        room_layout: (H, W, 4) uint8 array
        room_dims: Dict with room dimension info
    """
    # Get room bounds
    min_x, min_z, max_x, max_z = get_room_bounds(items)
    width = max_x - min_x
    depth = max_z - min_z
    
    # Calculate mask size
    mask_width = int(width * pixels_per_meter)
    mask_height = int(depth * pixels_per_meter)
    
    # Initialize 4-channel mask
    room_layout = np.zeros((mask_height, mask_width, 4), dtype=np.uint8)
    
    # Separate items by type
    floors = [item for item in items if item.get('diffusion_type') == 'floor']
    walls = [item for item in items if item.get('diffusion_type') == 'wall']
    doors = [item for item in items if item.get('diffusion_type') == 'door']
    windows = [item for item in items if item.get('diffusion_type') == 'window']
    
    # Draw each type to its channel
    # Channel 0: Floors
    for floor in floors:
        draw_floor_polygon(room_layout, floor, min_x, min_z, pixels_per_meter)
    
    # Channel 1: Walls
    for wall in walls:
        draw_architecture_item(room_layout, wall, min_x, min_z, pixels_per_meter, 
                               channel=1, thickness=3)
    
    # Channel 2: Doors
    for door in doors:
        draw_architecture_item(room_layout, door, min_x, min_z, pixels_per_meter, 
                               channel=2, thickness=5)
    
    # Channel 3: Windows
    for window in windows:
        draw_architecture_item(room_layout, window, min_x, min_z, pixels_per_meter, 
                               channel=3, thickness=5)
    
    # Resize to standard size (128x128)
    room_layout_resized = np.zeros((MASK_SIZE, MASK_SIZE, 4), dtype=np.uint8)
    for c in range(4):
        room_layout_resized[:, :, c] = cv2.resize(
            room_layout[:, :, c], 
            (MASK_SIZE, MASK_SIZE), 
            interpolation=cv2.INTER_NEAREST
        )
    
    # Room dimensions
    room_dims = {
        'width': width,
        'depth': depth,
        'min_x': min_x,
        'min_z': min_z,
        'max_x': max_x,
        'max_z': max_z,
        'pixels_per_meter': pixels_per_meter
    }
    
    return room_layout_resized, room_dims


def extract_furniture(items, room_dims, pixels_per_meter=PIXELS_PER_METER):
    """Extract furniture objects and normalize coordinates."""
    furniture_list = []
    
    min_x = room_dims['min_x']
    min_z = room_dims['min_z']
    width = room_dims['width']
    depth = room_dims['depth']
    
    for item in items:
        diff_type = item.get('diffusion_type', '')
        
        if diff_type not in FURNITURE_TYPES:
            continue
        
        # Extract position (normalized to [-1, 1])
        cx = item.get('center_x', 0)
        cz = item.get('center_z', 0)
        
        # Normalize to [0, 1] then to [-1, 1]
        norm_x = (cx - min_x) / width if width > 0 else 0
        norm_z = (cz - min_z) / depth if depth > 0 else 0
        norm_x = norm_x * 2 - 1
        norm_z = norm_z * 2 - 1
        
        # Extract size (normalized)
        sx = item.get('size_x', 0) / width if width > 0 else 0
        sz = item.get('size_z', 0) / depth if depth > 0 else 0
        
        # Extract angle (convert to radians, normalized to [0, 2π])
        yaw = item.get('yaw', 0)
        angle_rad = np.radians(yaw) % (2 * np.pi)
        
        furniture_list.append({
            'type': diff_type,
            'translation': [norm_x, norm_z],
            'size': [sx, sz],
            'angle': angle_rad
        })
    
    return furniture_list


def process_scene(json_path, output_dir):
    """Process a single scene JSON file."""
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    items = data.get('items', [])
    
    # Filter out skip types
    items = [item for item in items 
             if item.get('diffusion_type', '') not in SKIP_TYPES]
    
    # Check if scene has furniture
    furniture_count = sum(1 for item in items 
                         if item.get('diffusion_type', '') in FURNITURE_TYPES)
    if furniture_count == 0:
        return False
    
    # Create 4-channel room layout
    room_layout, room_dims = create_4channel_room_layout(items)
    
    # Extract furniture
    furniture = extract_furniture(items, room_dims)
    
    if len(furniture) == 0:
        return False
    
    # Convert to arrays
    N = len(furniture)
    class_labels = np.zeros((N, 6), dtype=np.float32)  # 4 furniture + 2 tokens
    translations = np.zeros((N, 2), dtype=np.float32)
    sizes = np.zeros((N, 2), dtype=np.float32)
    angles = np.zeros((N, 2), dtype=np.float32)  # [cos, sin]
    
    for i, furn in enumerate(furniture):
        # Class label (one-hot)
        class_idx = FURNITURE_TYPES.index(furn['type'])
        class_labels[i, class_idx] = 1.0
        
        # Translation
        translations[i] = furn['translation']
        
        # Size
        sizes[i] = furn['size']
        
        # Angle (as cos, sin)
        angle = furn['angle']
        angles[i] = [np.cos(angle), np.sin(angle)]
    
    # Create output directory
    scene_id = json_path.stem
    scene_output_dir = output_dir / scene_id
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save NPZ
    npz_path = scene_output_dir / 'boxes.npz'
    np.savez(
        npz_path,
        class_labels=class_labels,
        translations=translations,
        sizes=sizes,
        angles=angles,
        room_layout=room_layout,  # (128, 128, 4)
        objfeats_32=np.zeros((N, 32), dtype=np.float32)  # Placeholder
    )
    
    # Save metadata
    metadata = {
        'scene_id': scene_id,
        'num_furniture': N,
        'room_dims': {k: float(v) for k, v in room_dims.items()},
        'furniture_types': [furn['type'] for furn in furniture]
    }
    
    metadata_path = scene_output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save visualization
    save_visualization(room_layout, class_labels, translations, sizes, 
                      scene_output_dir / 'visualization.png')
    
    return True


def save_visualization(room_layout, class_labels, translations, sizes, output_path):
    """Save visualization of the 4-channel mask and furniture."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show each channel
    channel_names = ['Floor', 'Walls', 'Doors', 'Windows']
    for i in range(4):
        ax = axes[i // 2, i % 2]
        ax.imshow(room_layout[:, :, i], cmap='gray')
        ax.set_title(channel_names[i])
        ax.axis('off')
    
    # Combined view
    ax = axes[0, 2]
    combined = np.zeros((MASK_SIZE, MASK_SIZE, 3), dtype=np.uint8)
    combined[:, :, 0] = room_layout[:, :, 0]  # Floor - red channel
    combined[:, :, 1] = room_layout[:, :, 1]  # Walls - green channel
    combined[:, :, 2] = room_layout[:, :, 2] + room_layout[:, :, 3]  # Door+Window - blue
    ax.imshow(combined)
    ax.set_title('Combined (R=Floor, G=Wall, B=Door/Win)')
    ax.axis('off')
    
    # Furniture overlay
    ax = axes[1, 2]
    ax.imshow(room_layout[:, :, 0], cmap='gray', alpha=0.3)
    
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(len(class_labels)):
        class_idx = np.argmax(class_labels[i, :4])
        x = (translations[i, 0] + 1) * MASK_SIZE / 2
        z = (translations[i, 1] + 1) * MASK_SIZE / 2
        w = sizes[i, 0] * MASK_SIZE / 2
        h = sizes[i, 1] * MASK_SIZE / 2
        
        rect = plt.Rectangle((x - w/2, z - h/2), w, h,
                            fill=False, edgecolor=colors[class_idx], linewidth=2)
        ax.add_patch(rect)
        ax.text(x, z, FURNITURE_TYPES[class_idx], 
               color=colors[class_idx], fontsize=8)
    
    ax.set_title('Furniture Layout')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Preprocess bathroom scenes with 4-channel architecture')
    parser.add_argument('input_dir', type=str, help='Input directory with JSON files')
    parser.add_argument('output_dir', type=str, help='Output directory for processed data')
    parser.add_argument('--pixels-per-meter', type=int, default=40, 
                       help='Pixels per meter for mask rendering')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = list(input_dir.glob('*.json'))
    print(f"Found {len(json_files)} JSON files")
    
    # Process each file
    success_count = 0
    for json_file in tqdm(json_files, desc='Processing scenes'):
        try:
            if process_scene(json_file, output_dir):
                success_count += 1
        except Exception as e:
            print(f"\nError processing {json_file.name}: {e}")
            continue
    
    print(f"\n✓ Successfully processed {success_count}/{len(json_files)} scenes")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
