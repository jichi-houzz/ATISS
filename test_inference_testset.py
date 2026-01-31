"""
Test inference on test/validation set and save results to NPZ with visualizations
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from scene_synthesis.datasets.bathroom import BathroomDataset
from scene_synthesis.networks import build_network


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


def generate_from_test_set(network, config, split='test', num_scenes=10, device='cpu'):
    """Generate scenes from test set room masks."""
    if split == 'test':
        split_file = config['data']['test_split_file']
    elif split == 'val':
        split_file = config['data']['val_split_file']
    else:
        split_file = config['data']['train_split_file']
        #raise ValueError(f"Invalid split: {split}")

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

        room_mask = torch.from_numpy(scene.room_mask).permute(2, 0, 1)[None].float().to(device)

        with torch.no_grad():
            generated_boxes = network.generate_boxes(
                room_mask=room_mask,
                max_boxes=20,
                device=device
            )

        generated = {
            'class_labels': generated_boxes['class_labels'][0].cpu().numpy(),
            'translations': generated_boxes['translations'][0].cpu().numpy(),
            'sizes': generated_boxes['sizes'][0].cpu().numpy(),
            'angles': generated_boxes['angles'][0].cpu().numpy(),
            'room_mask': scene.room_mask[:, :, 0]
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
            gt_translations.append(bbox.centroid())
            gt_sizes.append(bbox.size)
            gt_angles.append([bbox.z_angle])

        ground_truth = {
            'class_labels': np.array(gt_class_labels, dtype=np.float32),
            'translations': np.array(gt_translations, dtype=np.float32),
            'sizes': np.array(gt_sizes, dtype=np.float32),
            'angles': np.array(gt_angles, dtype=np.float32),
            'room_mask': scene.room_mask[:, :, 0]
        }

        results.append((ground_truth, generated, scene_id))

    return results


def create_comparison_image(ground_truth, generated, scene_id, output_path):
    """Create side-by-side comparison visualization."""
    from PIL import Image, ImageDraw, ImageFont

    room_mask = ground_truth['room_mask']
    H, W = room_mask.shape

    mask_img = (room_mask * 255).astype(np.uint8)
    img_gt = Image.fromarray(mask_img).convert('RGB')
    img_gen = Image.fromarray(mask_img).convert('RGB')

    draw_gt = ImageDraw.Draw(img_gt)
    draw_gen = ImageDraw.Draw(img_gen)

    colors = {
        0: (255, 0, 0),      # toilet - red
        1: (0, 255, 0),      # vanity - green
        2: (0, 0, 255),      # shower - blue
        3: (255, 255, 0)     # tub - yellow
    }
    categories = ['toilet', 'vanity', 'shower', 'tub']

    # Draw ground truth
    gt_classes = ground_truth['class_labels']
    gt_trans = ground_truth['translations']
    gt_sizes = ground_truth['sizes']

    for i in range(len(gt_classes)):
        class_idx = np.argmax(gt_classes[i, :4])
        color = colors.get(class_idx, (128, 128, 128))

        x = int((gt_trans[i, 0] + 1) * W / 2)
        z = int((gt_trans[i, 2] + 1) * H / 2)
        w = int(abs(gt_sizes[i, 0]) * W / 2)
        d = int(abs(gt_sizes[i, 2]) * H / 2)

        x1, z1 = max(0, x - w//2), max(0, z - d//2)
        x2, z2 = min(W, x + w//2), min(H, z + d//2)

        if x1 < x2 and z1 < z2:
            draw_gt.rectangle([x1, z1, x2, z2], outline=color, width=3)
            draw_gt.text((x, z), categories[class_idx], fill=color)

    # Draw generated
    gen_classes = generated['class_labels']
    gen_trans = generated['translations']
    gen_sizes = generated['sizes']

    for i in range(len(gen_classes)):
        if gen_classes[i, -2] == 1 or gen_classes[i, -1] == 1:
            continue

        class_idx = np.argmax(gen_classes[i, :4])
        color = colors.get(class_idx, (128, 128, 128))

        x = int((gen_trans[i, 0] + 1) * W / 2)
        z = int((gen_trans[i, 2] + 1) * H / 2)
        w = int(abs(gen_sizes[i, 0]) * W / 2)
        d = int(abs(gen_sizes[i, 2]) * H / 2)

        x1, z1 = max(0, x - w//2), max(0, z - d//2)
        x2, z2 = min(W, x + w//2), min(H, z + d//2)

        if x1 < x2 and z1 < z2:
            draw_gen.rectangle([x1, z1, x2, z2], outline=color, width=3)
            label = categories[class_idx] if class_idx < 4 else "?"
            draw_gen.text((x, z), label, fill=color)

    # Combine images
    combined_width = W * 2 + 40
    combined_height = H + 80
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

    # Titles
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

    for gt, gen, scene_id in tqdm(results, desc="Saving"):
        gt_path = output_path / f"{scene_id}_gt.npz"
        np.savez(gt_path, **gt)

        gen_path = output_path / f"{scene_id}_generated.npz"
        np.savez(gen_path, **gen)

        vis_path = output_path / f"{scene_id}_comparison.png"
        create_comparison_image(gt, gen, scene_id, vis_path)

    print(f"\n✓ Saved {len(results)} scenes:")
    print(f"  - *_gt.npz: Ground truth data")
    print(f"  - *_generated.npz: Generated data")
    print(f"  - *_comparison.png: Visualization")


def print_comparison(ground_truth, generated, scene_id):
    """Print text comparison."""
    print(f"\n{'='*60}")
    print(f"Scene: {scene_id}")
    print(f"{'='*60}")

    gt_count = len(ground_truth['class_labels'])

    gen_count = 0
    for i in range(len(generated['class_labels'])):
        if generated['class_labels'][i, -1] == 1:
            break
        if generated['class_labels'][i, -2] != 1:
            gen_count += 1

    print(f"\nGround Truth: {gt_count} objects")
    print(f"Generated:    {gen_count} objects")

    categories = ['toilet', 'vanity', 'shower', 'tub']

    print(f"\nGround Truth Objects:")
    for i in range(len(ground_truth['class_labels'])):
        class_idx = np.argmax(ground_truth['class_labels'][i, :4])
        pos = ground_truth['translations'][i]
        size = ground_truth['sizes'][i]
        angle = ground_truth['angles'][i, 0]
        print(f"  {i+1}. {categories[class_idx]:8s} pos=[{pos[0]:6.3f}, {pos[2]:6.3f}] "
              f"size=[{size[0]:6.3f}, {size[2]:6.3f}] angle={angle:6.3f}")

    print(f"\nGenerated Objects:")
    for i in range(len(generated['class_labels'])):
        if generated['class_labels'][i, -1] == 1:
            print(f"  [END]")
            break
        if generated['class_labels'][i, -2] == 1:
            print(f"  [START]")
            continue

        class_idx = np.argmax(generated['class_labels'][i, :4])
        pos = generated['translations'][i]
        size = generated['sizes'][i]
        angle = generated['angles'][i, 0]
        print(f"  {i}. {categories[class_idx]:8s} pos=[{pos[0]:6.3f}, {pos[2]:6.3f}] "
              f"size=[{size[0]:6.3f}, {size[2]:6.3f}] angle={angle:6.3f}")


def main():
    parser = argparse.ArgumentParser(description='Test inference on test/val set with visualizations')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/bathrooms_test_config.yaml')
    parser.add_argument('--output-dir', type=str, default='inference_results')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'])
    parser.add_argument('--num-scenes', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    print("Loading model...")
    network, dataset, config = load_model(args.config, args.checkpoint, device='cpu')
    print(f"✓ Model loaded from: {args.checkpoint}")

    print(f"\nGenerating from {args.split} set...")
    results = generate_from_test_set(
        network, config, split=args.split, num_scenes=args.num_scenes, device='cpu',
        #split='test',
        #split='eval',
        split='train',
    )

    save_results_to_npz(results, args.output_dir)

    if args.verbose:
        for gt, gen, scene_id in results[:3]:
            print_comparison(gt, gen, scene_id)

    print(f"\n✓ Complete! Generated {len(results)} scenes")
    print(f"  Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
