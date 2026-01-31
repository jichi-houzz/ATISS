#!/usr/bin/env python3
"""
分析预处理后的 NPZ 数据分布
检查 translations, sizes, angles 的实际值
"""

import numpy as np
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt


def analyze_preprocessed_distribution(data_dir, output_dir=None):
    """
    分析预处理后的数据集分布
    
    Args:
        data_dir: 包含预处理 NPZ 文件的目录
        output_dir: 输出图表的目录（可选）
    """
    data_path = Path(data_dir)
    scene_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(scene_dirs)} scenes in {data_dir}\n")
    
    # 收集所有数据
    all_translations = []
    all_sizes = []
    all_angles = []
    all_categories = []
    
    category_map = {0: 'toilet', 1: 'vanity', 2: 'shower', 3: 'tub'}
    
    # 检查第一个场景的 metadata
    first_scene = list(scene_dirs)[0]
    metadata_path = first_scene / 'metadata.json'
    has_bounds = False
    bounds_info = None
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if 'bounds' in metadata:
                has_bounds = True
                bounds_info = metadata['bounds']
                print("Found bounds in metadata:")
                print(f"  {bounds_info}\n")
    
    for scene_dir in scene_dirs:
        npz_path = scene_dir / 'boxes.npz'
        
        if not npz_path.exists():
            continue
        
        try:
            data = np.load(npz_path)
            
            # 提取数据
            class_labels = data['class_labels']  # (N, 4) or (N, 6)
            translations = data['translations']  # (N, 2) or (N, 3)
            sizes = data['sizes']                # (N, 2) or (N, 3)
            angles = data['angles']              # (N, 2) - [cos, sin]
            
            N = len(class_labels)
            
            for i in range(N):
                # Get category
                category_idx = np.argmax(class_labels[i, :4])
                category = category_map.get(category_idx, 'unknown')
                
                # Extract translations (handle both 2D and 3D)
                if translations.shape[1] == 2:
                    trans = translations[i]  # (x, z)
                elif translations.shape[1] == 3:
                    trans = [translations[i, 0], translations[i, 2]]  # (x, z) from (x, y, z)
                else:
                    continue
                
                # Extract sizes (handle both 2D and 3D)
                if sizes.shape[1] == 2:
                    size = sizes[i]  # (width, depth)
                elif sizes.shape[1] == 3:
                    size = [sizes[i, 0], sizes[i, 2]]  # (width, depth) from (width, height, depth)
                else:
                    continue
                
                # Convert cos/sin to angle
                cos_angle, sin_angle = angles[i]
                angle_rad = np.arctan2(sin_angle, cos_angle)
                
                all_translations.append(trans)
                all_sizes.append(size)
                all_angles.append(angle_rad)
                all_categories.append(category)
        
        except Exception as e:
            print(f"Warning: Failed to load {scene_dir.name}: {e}")
            continue
    
    # 转换为 numpy 数组
    all_translations = np.array(all_translations)  # (N, 2)
    all_sizes = np.array(all_sizes)                # (N, 2)
    all_angles = np.array(all_angles)              # (N,)
    
    print("="*70)
    print("PREPROCESSED DATA DISTRIBUTION ANALYSIS")
    print("="*70)
    
    print(f"\nTotal furniture items: {len(all_categories)}")
    print(f"  toilet:  {all_categories.count('toilet')}")
    print(f"  vanity:  {all_categories.count('vanity')}")
    print(f"  shower:  {all_categories.count('shower')}")
    print(f"  tub:     {all_categories.count('tub')}")
    
    # Data format info
    print(f"\nData format:")
    print(f"  Translations shape: {all_translations.shape} (x, z)")
    print(f"  Sizes shape: {all_sizes.shape} (width, depth)")
    print(f"  Angles: stored as (cos, sin), converted to radians")
    
    # Translations 分布
    print("\n" + "="*70)
    print("TRANSLATIONS DISTRIBUTION")
    print("="*70)
    print(f"\nTranslation X:")
    print(f"  Min:        {all_translations[:, 0].min():10.4f}")
    print(f"  Max:        {all_translations[:, 0].max():10.4f}")
    print(f"  Mean:       {all_translations[:, 0].mean():10.4f}")
    print(f"  Median:     {np.median(all_translations[:, 0]):10.4f}")
    print(f"  Std:        {all_translations[:, 0].std():10.4f}")
    print(f"  Percentiles:")
    print(f"    1st:      {np.percentile(all_translations[:, 0], 1):10.4f}")
    print(f"    5th:      {np.percentile(all_translations[:, 0], 5):10.4f}")
    print(f"    95th:     {np.percentile(all_translations[:, 0], 95):10.4f}")
    print(f"    99th:     {np.percentile(all_translations[:, 0], 99):10.4f}")
    
    print(f"\nTranslation Z:")
    print(f"  Min:        {all_translations[:, 1].min():10.4f}")
    print(f"  Max:        {all_translations[:, 1].max():10.4f}")
    print(f"  Mean:       {all_translations[:, 1].mean():10.4f}")
    print(f"  Median:     {np.median(all_translations[:, 1]):10.4f}")
    print(f"  Std:        {all_translations[:, 1].std():10.4f}")
    print(f"  Percentiles:")
    print(f"    1st:      {np.percentile(all_translations[:, 1], 1):10.4f}")
    print(f"    5th:      {np.percentile(all_translations[:, 1], 5):10.4f}")
    print(f"    95th:     {np.percentile(all_translations[:, 1], 95):10.4f}")
    print(f"    99th:     {np.percentile(all_translations[:, 1], 99):10.4f}")
    
    # Sizes 分布
    print("\n" + "="*70)
    print("SIZES DISTRIBUTION")
    print("="*70)
    print(f"\nSize X (Width):")
    print(f"  Min:        {all_sizes[:, 0].min():10.4f}")
    print(f"  Max:        {all_sizes[:, 0].max():10.4f}")
    print(f"  Mean:       {all_sizes[:, 0].mean():10.4f}")
    print(f"  Median:     {np.median(all_sizes[:, 0]):10.4f}")
    print(f"  Std:        {all_sizes[:, 0].std():10.4f}")
    print(f"  Percentiles:")
    print(f"    1st:      {np.percentile(all_sizes[:, 0], 1):10.4f}")
    print(f"    5th:      {np.percentile(all_sizes[:, 0], 5):10.4f}")
    print(f"    95th:     {np.percentile(all_sizes[:, 0], 95):10.4f}")
    print(f"    99th:     {np.percentile(all_sizes[:, 0], 99):10.4f}")
    
    print(f"\nSize Z (Depth):")
    print(f"  Min:        {all_sizes[:, 1].min():10.4f}")
    print(f"  Max:        {all_sizes[:, 1].max():10.4f}")
    print(f"  Mean:       {all_sizes[:, 1].mean():10.4f}")
    print(f"  Median:     {np.median(all_sizes[:, 1]):10.4f}")
    print(f"  Std:        {all_sizes[:, 1].std():10.4f}")
    print(f"  Percentiles:")
    print(f"    1st:      {np.percentile(all_sizes[:, 1], 1):10.4f}")
    print(f"    5th:      {np.percentile(all_sizes[:, 1], 5):10.4f}")
    print(f"    95th:     {np.percentile(all_sizes[:, 1], 95):10.4f}")
    print(f"    99th:     {np.percentile(all_sizes[:, 1], 99):10.4f}")
    
    # Angles 分布
    print("\n" + "="*70)
    print("ANGLES DISTRIBUTION")
    print("="*70)
    all_angles_deg = np.degrees(all_angles)
    
    print(f"\nRadians:")
    print(f"  Min:        {all_angles.min():10.4f} rad")
    print(f"  Max:        {all_angles.max():10.4f} rad")
    print(f"  Mean:       {all_angles.mean():10.4f} rad")
    print(f"  Median:     {np.median(all_angles):10.4f} rad")
    print(f"  Std:        {all_angles.std():10.4f} rad")
    
    print(f"\nDegrees:")
    print(f"  Min:        {all_angles_deg.min():10.1f}°")
    print(f"  Max:        {all_angles_deg.max():10.1f}°")
    print(f"  Mean:       {all_angles_deg.mean():10.1f}°")
    print(f"  Median:     {np.median(all_angles_deg):10.1f}°")
    print(f"  Std:        {all_angles_deg.std():10.1f}°")
    
    # 检查数据范围
    print("\n" + "="*70)
    print("DATA RANGE CHECK")
    print("="*70)
    
    # Check if normalized
    trans_in_01 = (all_translations.min() >= 0) and (all_translations.max() <= 1)
    trans_in_neg11 = (all_translations.min() >= -1) and (all_translations.max() <= 1)
    size_in_01 = (all_sizes.min() >= 0) and (all_sizes.max() <= 1)
    
    print(f"\nTranslations:")
    if trans_in_01:
        print(f"  ✓ In range [0, 1] - Normalized to unit range")
    elif trans_in_neg11:
        print(f"  ✓ In range [-1, 1] - Normalized centered")
    else:
        print(f"  ✓ Raw values (meters) - Not normalized")
        print(f"    Range: [{all_translations.min():.2f}, {all_translations.max():.2f}]")
    
    print(f"\nSizes:")
    if size_in_01:
        print(f"  ✓ In range [0, 1] - Normalized")
    else:
        print(f"  ✓ Raw values (meters) - Not normalized")
        print(f"    Range: [{all_sizes.min():.2f}, {all_sizes.max():.2f}]")
    
    print(f"\nAngles:")
    print(f"  ✓ Stored as (cos, sin), range: [{all_angles.min():.2f}, {all_angles.max():.2f}] rad")
    
    # 如果有 bounds，检查一致性
    if has_bounds and bounds_info:
        print("\n" + "="*70)
        print("BOUNDS CONSISTENCY CHECK")
        print("="*70)
        print(f"\nMetadata bounds:")
        print(f"  Position: [{bounds_info['bounds_min']}, {bounds_info['bounds_max']}]")
        print(f"  Size: [{bounds_info['size_min']}, {bounds_info['size_max']}]")
        
        print(f"\nActual data range:")
        print(f"  Position: [{all_translations.min():.2f}, {all_translations.max():.2f}]")
        print(f"  Size: [{all_sizes.min():.2f}, {all_sizes.max():.2f}]")
        
        # 检查是否匹配
        trans_matches = (all_translations.min() >= 0) and (all_translations.max() <= 1)
        if trans_matches:
            print(f"  ✓ Data normalized to [0, 1] using bounds")
        else:
            print(f"  ⚠ Data does not match expected normalized range")
    
    # 绘图（如果指定了输出目录）
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n" + "="*70)
        print(f"Saving plots to {output_dir}/")
        print("="*70)
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Preprocessed NPZ Data Distribution', fontsize=16)
        
        # Translation X
        axes[0, 0].hist(all_translations[:, 0], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Translation X')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Translation X Distribution')
        axes[0, 0].axvline(all_translations[:, 0].mean(), color='r', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        # Translation Z
        axes[0, 1].hist(all_translations[:, 1], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Translation Z')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Translation Z Distribution')
        axes[0, 1].axvline(all_translations[:, 1].mean(), color='r', linestyle='--', label='Mean')
        axes[0, 1].legend()
        
        # Translation 2D
        axes[0, 2].scatter(all_translations[:, 0], all_translations[:, 1], alpha=0.3, s=10)
        axes[0, 2].set_xlabel('Translation X')
        axes[0, 2].set_ylabel('Translation Z')
        axes[0, 2].set_title('Translation 2D Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Size X
        axes[1, 0].hist(all_sizes[:, 0], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Size X')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Size X Distribution')
        axes[1, 0].axvline(all_sizes[:, 0].mean(), color='r', linestyle='--', label='Mean')
        axes[1, 0].legend()
        
        # Size Z
        axes[1, 1].hist(all_sizes[:, 1], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Size Z')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Size Z Distribution')
        axes[1, 1].axvline(all_sizes[:, 1].mean(), color='r', linestyle='--', label='Mean')
        axes[1, 1].legend()
        
        # Angles
        axes[1, 2].hist(all_angles_deg, bins=36, alpha=0.7, edgecolor='black')
        axes[1, 2].set_xlabel('Angle (degrees)')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Angle Distribution')
        axes[1, 2].axvline(0, color='r', linestyle='--', alpha=0.5)
        axes[1, 2].axvline(90, color='r', linestyle='--', alpha=0.5)
        axes[1, 2].axvline(180, color='r', linestyle='--', alpha=0.5)
        axes[1, 2].axvline(-90, color='r', linestyle='--', alpha=0.5)
        axes[1, 2].axvline(-180, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path / 'preprocessed_distribution.png', dpi=150)
        print(f"  ✓ Saved preprocessed_distribution.png")
        
        plt.close('all')
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze preprocessed NPZ data distribution')
    parser.add_argument('data_dir', type=str, help='Directory containing preprocessed NPZ files')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Directory to save plots (optional)')
    
    args = parser.parse_args()
    
    analyze_preprocessed_distribution(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
