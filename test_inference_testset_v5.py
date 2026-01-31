"""
Test inference with architecture visualization
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import time

from scene_synthesis.datasets.bathroom import BathroomDataset
from scene_synthesis.networks import build_network

USE_BATCH_GEN = False

def load_architecture_from_json(json_path, room_dims, bounds):
    """
    Load architecture items (walls, doors, windows) from original JSON.

    Args:
        json_path: Path to original JSON
        room_dims: Room dimensions dict
        bounds: Normalization bounds dict

    Returns:
        List of architecture dicts with normalized coordinates
    """
    if not Path(json_path).exists():
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)

    items = data.get('items', [])
    architecture = []

    bounds_min = bounds['position_min']
    bounds_max = bounds['position_max']
    size_min = bounds['size_min']
    size_max = bounds['size_max']

    for item in items:
        diff_type = item.get('diffusion_type', '')

        if diff_type not in ['wall', 'door', 'window']:
            continue

        # Get world coordinates
        cx = item['center_x']
        cz = item['center_z']
        sx = item['size_x']
        sz = item['size_z']
        yaw = -item.get('yaw', 0.0)

        # Normalize to [-1, 1] (same as ATISS Scale class)
        # Step 1: normalize to [0, 1]
        x_01 = (cx - bounds_min) / (bounds_max - bounds_min)
        z_01 = (cz - bounds_min) / (bounds_max - bounds_min)
        sx_01 = (sx - size_min) / (size_max - size_min)
        sz_01 = (sz - size_min) / (size_max - size_min)

        # Step 2: scale to [-1, 1]
        x_norm = 2 * x_01 - 1
        z_norm = 2 * z_01 - 1
        sx_norm = 2 * sx_01 - 1
        sz_norm = 2 * sz_01 - 1

        # Angle in radians
        yaw_rad = np.radians(yaw)

        architecture.append({
            'type': diff_type,
            'position': (x_norm, z_norm),
            'size': (sx_norm, sz_norm),
            'angle': yaw_rad
        })

    return architecture


def load_model(config_path, checkpoint_path, device='cpu'):
    """Load trained model from checkpoint."""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset = BathroomDataset.from_dataset_directory(
        config['data']['dataset_directory']
    )

    input_dims = len(dataset.class_labels) + 3 + 3 + 1

    network, _, _ = build_network(
        input_dims=input_dims,
        n_classes=len(dataset.class_labels),
        config=config,
        weight_file=None,
        device=device
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint)
    network.eval()

    return network, dataset, config


def generate_from_test_set(network, config, json_dir, split='test', num_scenes=10, device='cpu'):
    """Generate scenes from test set room masks."""
    if split == 'test':
        split_file = config['data']['test_split_file']
    elif split == 'val':
        split_file = config['data']['val_split_file']
    elif split == 'train':
        split_file = config['data']['train_split_file']
    else:
        raise ValueError(f"Invalid split: {split}")

    with open(split_file, 'r') as f:
        scene_ids = [line.strip() for line in f if line.strip()]

    test_dataset = BathroomDataset.from_dataset_directory(
        config['data']['dataset_directory'],
        scene_ids=scene_ids[:num_scenes]
    )

    print(f"Generating from {len(test_dataset)} {split} scenes")

    results = []

    for idx in tqdm(range(len(test_dataset)), desc=f"Generating {split} scenes"):
        scene = test_dataset[idx]
        scene_id = scene.scene_id

        # Handle both (H, W) and (H, W, C) formats
        room_mask_np = scene.room_mask
        if len(room_mask_np.shape) == 2:
            room_mask_np = room_mask_np[:, :, np.newaxis]

        room_mask = torch.from_numpy(room_mask_np).permute(2, 0, 1)[None].float().to(device)

        start_time = time.time()
        with torch.no_grad():
            generated_boxes = network.generate_boxes_with_callback(
                room_mask=room_mask,
                max_boxes=4, # 4 furniture
                device=device
            )

        stop_time = time.time()
        latency = stop_time - start_time
        print(f"Latency: {latency}s")

        generated = {
            'class_labels': generated_boxes['class_labels'][0].cpu().numpy(),
            'translations': generated_boxes['translations'][0].cpu().numpy(),
            'sizes': generated_boxes['sizes'][0].cpu().numpy(),
            'angles': generated_boxes['angles'][0].cpu().numpy(),
            'room_mask': scene.room_mask[:, :, 0] if len(scene.room_mask.shape) == 3 else scene.room_mask
        }

        # Define bounds (same as bathroom_v2.py)
        bounds = {
            'position_min': -25.0,
            'position_max': 25.0,
            'size_min': 0.0,
            'size_max': 7.0
        }

        gt_class_labels = []
        gt_translations = []
        gt_sizes = []
        gt_angles = []

        for bbox in scene.bboxes:
            one_hot = np.zeros(6, dtype=np.float32)
            class_idx = test_dataset.CATEGORIES.index(bbox.label)
            one_hot[class_idx] = 1.0
            gt_class_labels.append(one_hot)

            # Extract 2D coordinates (x, z) from 3D centroid
            centroid = bbox.centroid()
            if len(centroid) == 3:
                # 3D format: (x, y, z) → take (x, z)
                x_real, z_real = centroid[0], centroid[2]
            else:
                # Already 2D
                x_real, z_real = centroid[0], centroid[1]

            # Normalize to [-1, 1] (same as training data)
            x_01 = (x_real - bounds['position_min']) / (bounds['position_max'] - bounds['position_min'])
            z_01 = (z_real - bounds['position_min']) / (bounds['position_max'] - bounds['position_min'])
            x_norm = 2 * x_01 - 1
            z_norm = 2 * z_01 - 1
            gt_translations.append([x_norm, z_norm])

            # Extract 2D size (width, depth)
            bbox_size = bbox.size
            if len(bbox_size) == 3:
                # 3D format: (width, height, depth) → take (width, depth)
                w_real, d_real = bbox_size[0], bbox_size[2]
            else:
                # Already 2D
                w_real, d_real = bbox_size[0], bbox_size[1]

            # Normalize to [-1, 1]
            w_01 = (w_real - bounds['size_min']) / (bounds['size_max'] - bounds['size_min'])
            d_01 = (d_real - bounds['size_min']) / (bounds['size_max'] - bounds['size_min'])
            w_norm = 2 * w_01 - 1
            d_norm = 2 * d_01 - 1
            gt_sizes.append([w_norm, d_norm])

            gt_angles.append([bbox.z_angle])

        ground_truth = {
            'class_labels': np.array(gt_class_labels, dtype=np.float32),
            'translations': np.array(gt_translations, dtype=np.float32),
            'sizes': np.array(gt_sizes, dtype=np.float32),
            'angles': np.array(gt_angles, dtype=np.float32),
            'room_mask': scene.room_mask[:, :, 0] if len(scene.room_mask.shape) == 3 else scene.room_mask
        }

        # Load room_dims from metadata
        metadata_path = Path(config['data']['dataset_directory']) / scene_id / 'metadata.json'
        room_dims = None

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                room_dims = metadata.get('room_dims', None)

        # Try to load architecture from original JSON
        if json_dir and room_dims:
            json_path = Path(json_dir) / f"{scene_id}.json"
            architecture = load_architecture_from_json(json_path, room_dims, bounds)

            print(f"DEBUG: Loaded {len(architecture)} architecture items for {scene_id}")

        results.append((ground_truth, generated, scene_id, room_dims, bounds, architecture))

    return results


def draw_architecture(draw, architecture, room_dims, W, H, denormalize_func):
    """Draw architecture items on the image."""
    arch_colors = {
        'wall': (100, 100, 100),    # Gray
        'door': (150, 75, 0),        # Brown
        'window': (135, 206, 235)    # Sky blue
    }

    for arch in architecture:
        arch_type = arch['type']
        color = arch_colors.get(arch_type, (128, 128, 128))

        # Denormalize
        px, pz, pw, ph = denormalize_func(
            arch['position'][0], arch['position'][1],
            arch['size'][0], arch['size'][1]
        )

        pw = abs(pw)
        ph = abs(ph)

        angle = arch['angle']

        # Calculate 4 corners with rotation
        hw, hh = pw / 2, ph / 2
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        corners_local = [
            (-hw, -hh),
            ( hw, -hh),
            ( hw,  hh),
            (-hw,  hh)
        ]

        corners = []
        for cx, cy in corners_local:
            rx = cx * cos_a - cy * sin_a
            ry = cx * sin_a + cy * cos_a
            corners.append((int(px + rx), int(pz + ry)))

        valid = any(0 <= x < W and 0 <= y < H for x, y in corners)

        if valid and pw > 0 and ph > 0:
            if arch_type == 'wall':
                # Draw walls as outline
                draw.polygon(corners, outline=color, width=2)
            else:
                # Draw doors/windows as filled
                draw.polygon(corners, fill=color, outline=color)


def create_comparison_image(ground_truth, generated, scene_id, room_dims, bounds, architecture, output_path):
    """Create side-by-side comparison with architecture overlay."""
    from PIL import Image, ImageDraw, ImageFont

    room_mask = ground_truth['room_mask']
    H, W = room_mask.shape

    if room_dims is None:
        print(f"Warning: No room_dims for {scene_id}")
        width, depth = 1.0, 1.0
        min_x, min_z = -0.5, -0.5
        center_x, center_z = 0, 0
    else:
        width = room_dims.get('width', 1.0)
        depth = room_dims.get('depth', 1.0)
        min_x = room_dims.get('min_x', 0)
        min_z = room_dims.get('min_z', 0)
        center_x = room_dims.get('center_x', 0)
        center_z = room_dims.get('center_z', 0)

    def denormalize_and_to_pixel(norm_x, norm_z, norm_size_x, norm_size_z):
        """Convert normalized coords to pixel coords using bounds."""
        # ATISS normalizes to [-1, 1], not [0, 1]
        # Need to convert back: descale(x) = (x + 1) / 2 * (max - min) + min

        position_min = bounds['position_min']
        position_max = bounds['position_max']
        size_min = bounds['size_min']
        size_max = bounds['size_max']

        # Denormalize from [-1, 1] to world coords (meters)
        real_x = (norm_x + 1) / 2 * (position_max - position_min) + position_min
        real_z = (norm_z + 1) / 2 * (position_max - position_min) + position_min
        real_size_x = (norm_size_x + 1) / 2 * (size_max - size_min) + size_min
        real_size_z = (norm_size_z + 1) / 2 * (size_max - size_min) + size_min

        # Convert to pixels at original scale
        PIXELS_PER_METER = 40
        canvas_width = width * PIXELS_PER_METER
        canvas_height = depth * PIXELS_PER_METER

        px_canvas = (real_x - min_x) * PIXELS_PER_METER
        pz_canvas = (real_z - min_z) * PIXELS_PER_METER
        pw_canvas = real_size_x * PIXELS_PER_METER
        ph_canvas = real_size_z * PIXELS_PER_METER

        # Apply scaling
        target_size = int(W * 0.8)
        scale_factor = min(target_size / canvas_width, target_size / canvas_height)

        new_width = canvas_width * scale_factor
        new_height = canvas_height * scale_factor

        px_scaled = px_canvas * scale_factor
        pz_scaled = pz_canvas * scale_factor
        pw_scaled = pw_canvas * scale_factor
        ph_scaled = ph_canvas * scale_factor

        # Add padding
        offset_x = (W - new_width) / 2
        offset_y = (H - new_height) / 2

        return int(px_scaled + offset_x), int(pz_scaled + offset_y), int(pw_scaled), int(ph_scaled)

    mask_img = (room_mask * 255).astype(np.uint8)
    img_gt = Image.fromarray(mask_img).convert('RGB')
    img_gen = Image.fromarray(mask_img).convert('RGB')

    draw_gt = ImageDraw.Draw(img_gt)
    draw_gen = ImageDraw.Draw(img_gen)

    # Draw architecture on GT image
    draw_architecture(draw_gt, architecture, room_dims, W, H, denormalize_and_to_pixel)
    draw_architecture(draw_gen, architecture, room_dims, W, H, denormalize_and_to_pixel)

    # Furniture colors
    colors = {
        0: (255, 0, 0),      # toilet - red
        1: (0, 255, 0),      # vanity - green
        2: (0, 0, 255),      # shower - blue
        3: (255, 255, 0)     # tub - yellow
    }
    categories = ['toilet', 'vanity', 'shower', 'tub']

    # Draw ground truth furniture
    gt_classes = ground_truth['class_labels']
    gt_trans = ground_truth['translations']
    gt_sizes = ground_truth['sizes']
    gt_angles = ground_truth['angles']

    for i in range(len(gt_classes)):
        class_idx = np.argmax(gt_classes[i, :4])
        color = colors.get(class_idx, (128, 128, 128))

        px, pz, pw, ph = denormalize_and_to_pixel(
            gt_trans[i, 0], gt_trans[i, 1],  # ← 改成 [i, 1]，因为现在是 2D
            gt_sizes[i, 0], gt_sizes[i, 1]   # ← 改成 [i, 1]
        )

        pw = abs(pw)
        ph = abs(ph)
        angle = gt_angles[i, 0]

        hw, hh = pw / 2, ph / 2
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        corners = []
        for cx, cy in corners_local:
            rx = cx * cos_a - cy * sin_a
            ry = cx * sin_a + cy * cos_a
            corners.append((int(px + rx), int(pz + ry)))

        valid = any(0 <= x < W and 0 <= y < H for x, y in corners)
        if valid and pw > 0 and ph > 0:
            draw_gt.polygon(corners, outline=color, width=3)
            #draw_gt.text((px, pz), categories[class_idx], fill=color)

    # Draw generated furniture
    gen_classes = generated['class_labels']
    gen_trans = generated['translations']
    gen_sizes = generated['sizes']
    gen_angles = generated['angles']

    for i in range(len(gen_classes)):
        if gen_classes[i, -2] == 1 or gen_classes[i, -1] == 1:
            continue

        class_idx = np.argmax(gen_classes[i, :4])
        color = colors.get(class_idx, (128, 128, 128))

        px, pz, pw, ph = denormalize_and_to_pixel(
            gen_trans[i, 0], gen_trans[i, 2],  # Use x and z (skip y at index 1)
            gen_sizes[i, 0], gen_sizes[i, 2]   # Use width and depth (skip height at index 1)
        )

        pw = abs(pw)
        ph = abs(ph)
        angle_norm = gen_angles[i, 0] if len(gen_angles[i].shape) > 0 else gen_angles[i]

        # Denormalize angle from [-1, 1] to [-π, π]
        angle = angle_norm * np.pi

        hw, hh = pw / 2, ph / 2
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        corners = []
        for cx, cy in corners_local:
            rx = cx * cos_a - cy * sin_a
            ry = cx * sin_a + cy * cos_a
            corners.append((int(px + rx), int(pz + ry)))

        valid = any(0 <= x < W and 0 <= y < H for x, y in corners)
        if valid and pw > 0 and ph > 0:
            draw_gen.polygon(corners, outline=color, width=3)
            #label = categories[class_idx] if class_idx < 4 else "?"
            #draw_gen.text((px, pz), label, fill=color)

    # Combine images
    combined_width = W * 2 + int(40*8)
    combined_height = H + int(80*1.5)
    combined = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

    combined.paste(img_gt, (10, 50))
    combined.paste(img_gen, (W + 30, 50))

    draw_combined = ImageDraw.Draw(combined)

    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()

    title = f"{scene_id[:60]}"
    draw_combined.text((10, 10), title, fill=(0, 0, 0), font=font_title)
    draw_combined.text((10 + W//2 - 40, 32), "Ground Truth", fill=(0, 0, 0), font=font_label)
    draw_combined.text((W + 30 + W//2 - 30, 32), "Generated", fill=(0, 0, 0), font=font_label)

    # Legend
    legend_y = H + 58
    for idx, category in enumerate(categories):
        color = colors[idx]
        legend_x = 10 + idx * 120
        draw_combined.rectangle([legend_x, legend_y, legend_x + 12, legend_y + 12],
                                fill=color, outline=(0, 0, 0))
        draw_combined.text((legend_x + 18, legend_y - 2), category, fill=(0, 0, 0), font=font_label)

    combined.save(output_path)


def save_results_to_npz(results, output_dir):
    """Save results to NPZ files and visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to {output_dir}/")

    for gt, gen, scene_id, room_dims, bounds, architecture in tqdm(results, desc="Saving"):
        gt_path = output_path / f"{scene_id}_gt.npz"
        np.savez(gt_path, **gt)

        gen_path = output_path / f"{scene_id}_generated.npz"
        np.savez(gen_path, **gen)

        vis_path = output_path / f"{scene_id}_comparison.png"
        create_comparison_image(gt, gen, scene_id, room_dims, bounds, architecture, vis_path)

    print(f"\n✓ Saved {len(results)} scenes")


def print_comparison(ground_truth, generated, scene_id, bounds=None):
    """Print text comparison with detailed object information."""
    print(f"\n{'='*60}")
    print(f"Scene: {scene_id}")
    print(f"{'='*60}")

    categories = ['toilet', 'vanity', 'shower', 'tub']

    # Ground truth count
    gt_count = len(ground_truth['class_labels'])
    print(f"\n📍 Ground Truth: {gt_count} objects")
    for i in range(gt_count):
        class_idx = np.argmax(ground_truth['class_labels'][i, :4])
        category = categories[class_idx]
        pos = ground_truth['translations'][i]
        size = ground_truth['sizes'][i]
        angle = ground_truth['angles'][i, 0] if len(ground_truth['angles'][i].shape) > 0 else ground_truth['angles'][i]

        # Handle both 2D (x, z) and 3D (x, y, z) formats
        if len(pos) == 3:
            # 3D format: use x and z, skip y
            print(f"  [{i}] {category:8s} | pos=({pos[0]:6.3f}, {pos[2]:6.3f}) | size=({size[0]:.3f}, {size[2]:.3f}) | angle={np.degrees(angle):6.1f}°")
        else:
            # 2D format: use as-is
            print(f"  [{i}] {category:8s} | pos=({pos[0]:6.3f}, {pos[1]:6.3f}) | size=({size[0]:.3f}, {size[1]:.3f}) | angle={np.degrees(angle):6.1f}°")

    # Generated objects
    gen_count = 0
    print(f"\n🎲 Generated Objects:")
    for i in range(len(generated['class_labels'])):
        # Skip start/end tokens
        if generated['class_labels'][i, -2] == 1:  # start token
            continue
        if generated['class_labels'][i, -1] == 1:  # end token
            print(f"  [{i}] END_TOKEN")
            break

        class_idx = np.argmax(generated['class_labels'][i, :4])
        category = categories[class_idx] if class_idx < 4 else "unknown"
        pos = generated['translations'][i]  # 3D: (x, y, z)
        size = generated['sizes'][i]        # 3D: (width, height, depth)
        angle_norm = generated['angles'][i, 0] if len(generated['angles'][i].shape) > 0 else generated['angles'][i]

        # Denormalize if bounds available
        if bounds:
            position_min = bounds['position_min']
            position_max = bounds['position_max']
            size_min = bounds['size_min']
            size_max = bounds['size_max']

            # ATISS uses [-1, 1] range, need to convert back
            # Use pos[0] (x) and pos[2] (z), skip pos[1] (y)
            real_x = (pos[0] + 1) / 2 * (position_max - position_min) + position_min
            real_z = (pos[2] + 1) / 2 * (position_max - position_min) + position_min
            real_size_x = (size[0] + 1) / 2 * (size_max - size_min) + size_min
            real_size_z = (size[2] + 1) / 2 * (size_max - size_min) + size_min

            # Denormalize angle from [-1, 1] to [-π, π]
            angle_real = (angle_norm + 1) / 2 * (np.pi - (-np.pi)) + (-np.pi)
            # Simplifies to: angle_real = angle_norm * π

            print(f"  [{i}] {category:8s} | pos=({pos[0]:6.3f}, {pos[2]:6.3f}) → ({real_x:6.2f}m, {real_z:6.2f}m) | "
                  f"size=({size[0]:.3f}, {size[2]:.3f}) → ({real_size_x:.2f}m, {real_size_z:.2f}m) | angle={np.degrees(angle_real):6.1f}°")
        else:
            # Without bounds, assume angle is already in radians
            print(f"  [{i}] {category:8s} | pos=({pos[0]:6.3f}, {pos[2]:6.3f}) | size=({size[0]:.3f}, {size[2]:.3f}) | angle={np.degrees(angle_norm):6.1f}°")

        gen_count += 1

    print(f"\n📊 Summary: GT={gt_count} objects, Generated={gen_count} objects")



def main():
    parser = argparse.ArgumentParser(description='Test inference with architecture visualization')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/bathrooms_test_config_4ch.yaml')
    parser.add_argument('--json-dir', type=str, default='data/bathroom_2.2k_filter',
                       help='Directory with original JSON files for architecture')
    parser.add_argument('--output-dir', type=str, default='inference_results')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'train'])
    parser.add_argument('--num-scenes', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    print("Loading model...")
    network, dataset, config = load_model(args.config, args.checkpoint, device='cpu')
    print(f"✓ Model loaded from: {args.checkpoint}")

    print(f"\nGenerating from {args.split} set...")
    results = generate_from_test_set(
        network, config, args.json_dir,
        split=args.split, num_scenes=args.num_scenes, device='cpu'
    )

    save_results_to_npz(results, args.output_dir)

    if args.verbose:
        for gt, gen, scene_id, room_dims, bounds, arch in results[:3]:
            print_comparison(gt, gen, scene_id, bounds)

    print(f"\n✓ Complete! Generated {len(results)} scenes")


if __name__ == "__main__":
    main()
