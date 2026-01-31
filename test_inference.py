"""
Test inference with trained bathroom model
"""

import torch
import numpy as np
from pathlib import Path
import argparse

from scene_synthesis.datasets.bathroom import BathroomDataset
from scene_synthesis.networks import build_network


def load_model(config_path, checkpoint_path, device='cpu'):
    """Load trained model from checkpoint."""
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset to get bounds
    dataset = BathroomDataset.from_dataset_directory(
        config['data']['dataset_directory']
    )
    
    # Build network
    # Calculate input_dims: n_classes + 3 (translation) + 3 (size) + 1 (angle)
    input_dims = len(dataset.class_labels) + 3 + 3 + 1
    
    network, _, _ = build_network(
        input_dims=input_dims,
        n_classes=len(dataset.class_labels),
        config=config,  # Pass full config, not just config["network"]
        weight_file=None,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Checkpoint is directly the state_dict, not a dictionary with 'model_state_dict' key
    network.load_state_dict(checkpoint)
    network.eval()
    
    return network, dataset, config


def generate_scene(network, dataset, device='cpu', max_boxes=10):
    """Generate a new scene from scratch."""
    
    # Get a random room mask from validation set
    scene = dataset[0]
    room_mask = torch.from_numpy(scene.room_mask).permute(2, 0, 1)[None].float().to(device)
    
    print(f"Generating scene with room mask shape: {room_mask.shape}")
    
    # Generate boxes
    with torch.no_grad():
        boxes = network.generate_boxes(
            room_mask=room_mask,
            max_boxes=max_boxes,
            device=device
        )
    
    return boxes, room_mask


def print_generated_scene(boxes, dataset):
    """Print generated scene in readable format."""
    class_labels = boxes['class_labels'][0].cpu().numpy()  # (N, 6)
    translations = boxes['translations'][0].cpu().numpy()  # (N, 3)
    sizes = boxes['sizes'][0].cpu().numpy()                # (N, 3)
    angles = boxes['angles'][0].cpu().numpy()              # (N, 1)
    
    print("\n=== Generated Scene ===")
    print(f"Total objects: {len(class_labels)}")
    
    for i in range(len(class_labels)):
        # Get class (ignore start/end tokens)
        class_idx = np.argmax(class_labels[i, :4])  # Only first 4 classes
        class_name = dataset.CATEGORIES[class_idx] if class_idx < 4 else "unknown"
        
        # Check if it's end token
        if class_labels[i, -1] == 1:  # End token
            print(f"\n{i}: [END TOKEN]")
            break
        
        # Check if it's start token
        if class_labels[i, -2] == 1:  # Start token
            print(f"{i}: [START TOKEN]")
            continue
        
        pos = translations[i]
        size = sizes[i]
        angle = angles[i, 0] if len(angles[i].shape) > 0 else angles[i]
        
        print(f"\n{i}: {class_name}")
        print(f"   Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        print(f"   Size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")
        print(f"   Angle: {angle:.3f} rad ({np.degrees(angle):.1f}°)")


def save_scene_visualization(boxes, room_mask, output_path):
    """Save a simple visualization of the generated scene."""
    from PIL import Image, ImageDraw
    
    # Get room mask as image
    mask = room_mask[0, 0].cpu().numpy()  # (H, W)
    mask_img = (mask * 255).astype(np.uint8)
    
    # Create RGB image
    img = Image.fromarray(mask_img).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Draw generated boxes
    class_labels = boxes['class_labels'][0].cpu().numpy()
    translations = boxes['translations'][0].cpu().numpy()
    sizes = boxes['sizes'][0].cpu().numpy()
    
    # Colors for different classes
    colors = {
        0: (255, 0, 0),    # toilet - red
        1: (0, 255, 0),    # vanity - green
        2: (0, 0, 255),    # shower - blue
        3: (255, 255, 0)   # tub - yellow
    }
    
    H, W = mask.shape
    for i in range(len(class_labels)):
        # Skip start/end tokens
        if class_labels[i, -2] == 1 or class_labels[i, -1] == 1:
            continue
        
        class_idx = np.argmax(class_labels[i, :4])
        
        # Convert normalized coordinates to pixel coordinates
        # translations are in [-1, 1]
        x = int((translations[i, 0] + 1) * W / 2)
        z = int((translations[i, 2] + 1) * H / 2)
        
        # Convert normalized size to pixels (take absolute value)
        w = int(abs(sizes[i, 0]) * W / 2)
        d = int(abs(sizes[i, 2]) * H / 2)
        
        # Draw bounding box (ensure x1 < x2 and z1 < z2)
        color = colors.get(class_idx, (128, 128, 128))
        x1, z1 = max(0, x - w//2), max(0, z - d//2)
        x2, z2 = min(W, x + w//2), min(H, z + d//2)
        
        # Swap if needed
        if x1 > x2:
            x1, x2 = x2, x1
        if z1 > z2:
            z1, z2 = z2, z1
        
        # Skip if box is still invalid
        if x1 >= x2 or z1 >= z2:
            print(f"  Warning: Skipping invalid box at index {i}")
            continue
        
        draw.rectangle([x1, z1, x2, z2], outline=color, width=2)
        draw.text((x, z), str(class_idx), fill=color)
    
    img.save(output_path)
    print(f"\nVisualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test inference with trained model')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/bathrooms_test_config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default='generated_scene.png',
                        help='Output visualization path')
    parser.add_argument('--max-boxes', type=int, default=10,
                        help='Maximum number of boxes to generate')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    network, dataset, config = load_model(
        args.config,
        args.checkpoint,
        device='cpu'
    )
    print(f"Model loaded from: {args.checkpoint}")
    print(f"Dataset: {len(dataset)} scenes")
    print(f"Classes: {dataset.class_labels}")
    
    # Generate scene
    print("\nGenerating scene...")
    boxes, room_mask = generate_scene(
        network,
        dataset,
        device='cpu',
        max_boxes=args.max_boxes
    )
    
    # Print results
    print_generated_scene(boxes, dataset)
    
    # Save visualization
    save_scene_visualization(boxes, room_mask, args.output)
    
    print("\n✓ Inference test complete!")


if __name__ == "__main__":
    main()
