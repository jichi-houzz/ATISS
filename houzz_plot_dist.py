#!/usr/bin/env python3
"""
分析 Bathroom 数据集的分布
打印 position, size, angle 的统计信息
"""

import json
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


def analyze_distribution(json_dir, output_dir=None):
    """
    分析数据集分布
    
    Args:
        json_dir: 包含 JSON 文件的目录
        output_dir: 输出图表的目录（可选）
    """
    json_path = Path(json_dir)
    json_files = list(json_path.glob('*.json'))
    
    print(f"Found {len(json_files)} JSON files in {json_dir}\n")
    
    # 收集所有数据
    all_positions_x = []
    all_positions_z = []
    all_sizes_x = []
    all_sizes_z = []
    all_angles = []
    all_categories = []
    
    furniture_types = ['toilet', 'vanity', 'shower', 'tub']
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            items = data.get('items', [])
            
            for item in items:
                diff_type = item.get('diffusion_type', '')
                
                if diff_type in furniture_types:
                    all_positions_x.append(item['center_x'])
                    all_positions_z.append(item['center_z'])
                    all_sizes_x.append(item['size_x'])
                    all_sizes_z.append(item['size_z'])
                    all_angles.append(item.get('yaw', 0.0))
                    all_categories.append(diff_type)
        
        except Exception as e:
            print(f"Warning: Failed to load {json_file.name}: {e}")
            continue
    
    # 转换为 numpy 数组
    all_positions_x = np.array(all_positions_x)
    all_positions_z = np.array(all_positions_z)
    all_sizes_x = np.array(all_sizes_x)
    all_sizes_z = np.array(all_sizes_z)
    all_angles = np.array(all_angles)
    
    print("="*70)
    print("DATASET DISTRIBUTION ANALYSIS")
    print("="*70)
    
    print(f"\nTotal furniture items: {len(all_categories)}")
    print(f"  toilet:  {all_categories.count('toilet')}")
    print(f"  vanity:  {all_categories.count('vanity')}")
    print(f"  shower:  {all_categories.count('shower')}")
    print(f"  tub:     {all_categories.count('tub')}")
    
    # Position 分布
    print("\n" + "="*70)
    print("POSITION DISTRIBUTION (meters)")
    print("="*70)
    print(f"\nPosition X:")
    print(f"  Min:        {all_positions_x.min():8.3f} m")
    print(f"  Max:        {all_positions_x.max():8.3f} m")
    print(f"  Mean:       {all_positions_x.mean():8.3f} m")
    print(f"  Median:     {np.median(all_positions_x):8.3f} m")
    print(f"  Std:        {all_positions_x.std():8.3f} m")
    print(f"  Percentiles:")
    print(f"    5th:      {np.percentile(all_positions_x, 5):8.3f} m")
    print(f"    95th:     {np.percentile(all_positions_x, 95):8.3f} m")
    print(f"    99th:     {np.percentile(all_positions_x, 99):8.3f} m")
    
    print(f"\nPosition Z:")
    print(f"  Min:        {all_positions_z.min():8.3f} m")
    print(f"  Max:        {all_positions_z.max():8.3f} m")
    print(f"  Mean:       {all_positions_z.mean():8.3f} m")
    print(f"  Median:     {np.median(all_positions_z):8.3f} m")
    print(f"  Std:        {all_positions_z.std():8.3f} m")
    print(f"  Percentiles:")
    print(f"    5th:      {np.percentile(all_positions_z, 5):8.3f} m")
    print(f"    95th:     {np.percentile(all_positions_z, 95):8.3f} m")
    print(f"    99th:     {np.percentile(all_positions_z, 99):8.3f} m")
    
    # Size 分布
    print("\n" + "="*70)
    print("SIZE DISTRIBUTION (meters)")
    print("="*70)
    print(f"\nSize X (Width):")
    print(f"  Min:        {all_sizes_x.min():8.3f} m")
    print(f"  Max:        {all_sizes_x.max():8.3f} m")
    print(f"  Mean:       {all_sizes_x.mean():8.3f} m")
    print(f"  Median:     {np.median(all_sizes_x):8.3f} m")
    print(f"  Std:        {all_sizes_x.std():8.3f} m")
    print(f"  Percentiles:")
    print(f"    5th:      {np.percentile(all_sizes_x, 5):8.3f} m")
    print(f"    95th:     {np.percentile(all_sizes_x, 95):8.3f} m")
    print(f"    99th:     {np.percentile(all_sizes_x, 99):8.3f} m")
    
    print(f"\nSize Z (Depth):")
    print(f"  Min:        {all_sizes_z.min():8.3f} m")
    print(f"  Max:        {all_sizes_z.max():8.3f} m")
    print(f"  Mean:       {all_sizes_z.mean():8.3f} m")
    print(f"  Median:     {np.median(all_sizes_z):8.3f} m")
    print(f"  Std:        {all_sizes_z.std():8.3f} m")
    print(f"  Percentiles:")
    print(f"    5th:      {np.percentile(all_sizes_z, 5):8.3f} m")
    print(f"    95th:     {np.percentile(all_sizes_z, 95):8.3f} m")
    print(f"    99th:     {np.percentile(all_sizes_z, 99):8.3f} m")
    
    # Angle 分布
    print("\n" + "="*70)
    print("ANGLE DISTRIBUTION")
    print("="*70)
    
    # JSON 中存储的是度数
    all_angles_deg = np.array(all_angles)  # 已经是度数
    
    print(f"\nRaw values (degrees from JSON):")
    print(f"  Min:        {all_angles_deg.min():8.1f}°")
    print(f"  Max:        {all_angles_deg.max():8.1f}°")
    print(f"  Mean:       {all_angles_deg.mean():8.1f}°")
    print(f"  Median:     {np.median(all_angles_deg):8.1f}°")
    print(f"  Std:        {all_angles_deg.std():8.1f}°")
    print(f"  Percentiles:")
    print(f"    5th:      {np.percentile(all_angles_deg, 5):8.1f}°")
    print(f"    95th:     {np.percentile(all_angles_deg, 95):8.1f}°")
    print(f"    99th:     {np.percentile(all_angles_deg, 99):8.1f}°")
    
    # 转换为弧度（ATISS 使用）
    all_angles_rad = np.radians(all_angles_deg)
    print(f"\nConverted to radians (for ATISS):")
    print(f"  Min:        {all_angles_rad.min():8.4f} rad")
    print(f"  Max:        {all_angles_rad.max():8.4f} rad")
    print(f"  Mean:       {all_angles_rad.mean():8.4f} rad")
    print(f"  Median:     {np.median(all_angles_rad):8.4f} rad")
    print(f"  Std:        {all_angles_rad.std():8.4f} rad")
    
    # 统计常见角度
    angle_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    angle_counts, _ = np.histogram(np.abs(all_angles_deg) % 360, bins=angle_bins)
    print(f"\nAngle distribution (absolute):")
    for i in range(len(angle_bins)-1):
        print(f"  {angle_bins[i]:3.0f}° - {angle_bins[i+1]:3.0f}°: {angle_counts[i]:5d} ({angle_counts[i]/len(all_angles)*100:5.1f}%)")
    
    # 推荐的 bounds
    print("\n" + "="*70)
    print("RECOMMENDED BOUNDS")
    print("="*70)
    
    # 使用 min/max
    print("\nOption 1: Use min/max (tightest)")
    print(f"  POSITION_MIN = {min(all_positions_x.min(), all_positions_z.min()):.1f}")
    print(f"  POSITION_MAX = {max(all_positions_x.max(), all_positions_z.max()):.1f}")
    print(f"  SIZE_MIN = {min(all_sizes_x.min(), all_sizes_z.min()):.3f}")
    print(f"  SIZE_MAX = {max(all_sizes_x.max(), all_sizes_z.max()):.1f}")
    
    # 使用 99th percentile (更安全)
    print("\nOption 2: Use 99th percentile (safer, handles outliers)")
    pos_99 = max(np.percentile(np.abs(all_positions_x), 99), 
                 np.percentile(np.abs(all_positions_z), 99))
    size_99 = max(np.percentile(all_sizes_x, 99), 
                  np.percentile(all_sizes_z, 99))
    print(f"  POSITION_MIN = -{pos_99:.1f}")
    print(f"  POSITION_MAX = {pos_99:.1f}")
    print(f"  SIZE_MIN = 0.0")
    print(f"  SIZE_MAX = {size_99:.1f}")
    
    # 使用整数 bounds (简洁)
    print("\nOption 3: Use round numbers (simplest)")
    pos_bound = np.ceil(pos_99 / 5) * 5  # 向上取整到最近的 5
    size_bound = np.ceil(size_99)  # 向上取整到整数
    print(f"  POSITION_MIN = -{pos_bound:.0f}")
    print(f"  POSITION_MAX = {pos_bound:.0f}")
    print(f"  SIZE_MIN = 0")
    print(f"  SIZE_MAX = {size_bound:.0f}")
    
    # 绘图（如果指定了输出目录）
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n" + "="*70)
        print(f"Saving plots to {output_dir}/")
        print("="*70)
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Bathroom Dataset Distribution', fontsize=16)
        
        # Position X
        axes[0, 0].hist(all_positions_x, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Position X (m)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Position X Distribution')
        axes[0, 0].axvline(all_positions_x.mean(), color='r', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        # Position Z
        axes[0, 1].hist(all_positions_z, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Position Z (m)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Position Z Distribution')
        axes[0, 1].axvline(all_positions_z.mean(), color='r', linestyle='--', label='Mean')
        axes[0, 1].legend()
        
        # Position 2D
        axes[0, 2].scatter(all_positions_x, all_positions_z, alpha=0.3, s=10)
        axes[0, 2].set_xlabel('Position X (m)')
        axes[0, 2].set_ylabel('Position Z (m)')
        axes[0, 2].set_title('Position 2D Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Size X
        axes[1, 0].hist(all_sizes_x, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Size X (m)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Size X Distribution')
        axes[1, 0].axvline(all_sizes_x.mean(), color='r', linestyle='--', label='Mean')
        axes[1, 0].legend()
        
        # Size Z
        axes[1, 1].hist(all_sizes_z, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Size Z (m)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Size Z Distribution')
        axes[1, 1].axvline(all_sizes_z.mean(), color='r', linestyle='--', label='Mean')
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
        
        plt.tight_layout()
        plt.savefig(output_path / 'distribution.png', dpi=150)
        print(f"  ✓ Saved distribution.png")
        
        # 按类别的分布
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
        fig2.suptitle('Distribution by Category', fontsize=16)
        
        for idx, category in enumerate(furniture_types):
            row = idx // 2
            col = idx % 2
            
            mask = np.array([c == category for c in all_categories])
            
            if mask.sum() > 0:
                axes2[row, col].scatter(
                    all_positions_x[mask], 
                    all_positions_z[mask], 
                    alpha=0.5, 
                    s=all_sizes_x[mask] * 100
                )
                axes2[row, col].set_xlabel('Position X (m)')
                axes2[row, col].set_ylabel('Position Z (m)')
                axes2[row, col].set_title(f'{category.capitalize()} (n={mask.sum()})')
                axes2[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'distribution_by_category.png', dpi=150)
        print(f"  ✓ Saved distribution_by_category.png")
        
        plt.close('all')
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze bathroom dataset distribution')
    parser.add_argument('json_dir', type=str, help='Directory containing JSON files')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Directory to save plots (optional)')
    
    args = parser.parse_args()
    
    analyze_distribution(args.json_dir, args.output_dir)


if __name__ == "__main__":
    main()
