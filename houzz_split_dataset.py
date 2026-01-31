"""
Split bathroom dataset into train/val/test sets
"""

import argparse
import numpy as np
from pathlib import Path


def split_dataset(dataset_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train/val/test sets.
    
    Args:
        dataset_dir: Path to preprocessed dataset directory
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for testing (default: 0.15)
        seed: Random seed for reproducibility
    """
    dataset_path = Path(dataset_dir)
    
    # Get all scene directories
    scene_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    scene_ids = [d.name for d in scene_dirs]
    
    print(f"Found {len(scene_ids)} scenes in {dataset_dir}")
    
    # Verify ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Shuffle with fixed seed
    np.random.seed(seed)
    shuffled_ids = np.random.permutation(scene_ids)
    
    # Calculate split indices
    n_total = len(shuffled_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split
    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]
    
    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_ids)} scenes ({len(train_ids)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_ids)} scenes ({len(val_ids)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_ids)} scenes ({len(test_ids)/n_total*100:.1f}%)")
    
    # Save splits to text files
    output_dir = dataset_path.parent if dataset_path.parent.exists() else Path('.')
    
    train_file = output_dir / 'train_scenes.txt'
    val_file = output_dir / 'val_scenes.txt'
    test_file = output_dir / 'test_scenes.txt'
    
    # Write train
    with open(train_file, 'w') as f:
        for scene_id in sorted(train_ids):
            f.write(f"{scene_id}\n")
    
    # Write val
    with open(val_file, 'w') as f:
        for scene_id in sorted(val_ids):
            f.write(f"{scene_id}\n")
    
    # Write test
    with open(test_file, 'w') as f:
        for scene_id in sorted(test_ids):
            f.write(f"{scene_id}\n")
    
    print(f"\nSplit files saved:")
    print(f"  Train: {train_file}")
    print(f"  Val:   {val_file}")
    print(f"  Test:  {test_file}")
    
    return train_ids, val_ids, test_ids


def load_split(split_file):
    """
    Load scene IDs from split file.
    
    Args:
        split_file: Path to text file with one scene ID per line
        
    Returns:
        List of scene IDs
    """
    with open(split_file, 'r') as f:
        scene_ids = [line.strip() for line in f if line.strip()]
    return scene_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split bathroom dataset into train/val/test')
    parser.add_argument('dataset_dir', type=str, help='Path to preprocessed dataset directory')
    parser.add_argument('--train', type=float, default=0.7, help='Train ratio (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.15, help='Val ratio (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.15, help='Test ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    split_dataset(
        args.dataset_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )
