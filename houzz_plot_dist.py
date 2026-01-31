#!/usr/bin/env python3
"""
统计数据集的房间和家具尺寸，用于确定全局归一化参数
"""

import numpy as np
from pathlib import Path
import json

# 路径
preprocessed_dir = Path('data/preprocessed_4ch')
json_dir = Path('data/bathroom_2.2k_filter')

print("=== Collecting data ===")

# 1. 统计房间尺寸
all_room_widths = []
all_room_depths = []

for scene_dir in preprocessed_dir.iterdir():
    if not scene_dir.is_dir():
        continue

    meta_path = scene_dir / 'metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            room_dims = meta.get('room_dims', {})
            if 'width' in room_dims:
                all_room_widths.append(room_dims['width'])
                all_room_depths.append(room_dims['depth'])

print(f"Found {len(all_room_widths)} rooms in preprocessed data")

# 2. 统计家具真实尺寸（从原始 JSON）
all_furniture_sizes_x = []
all_furniture_sizes_z = []

furniture_types = ['toilet', 'vanity', 'shower', 'tub']

if json_dir.exists():
    json_files = list(json_dir.glob('*.json'))
    print(f"Processing {len(json_files)} JSON files...")

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
                items = data.get('items', [])

                for item in items:
                    diff_type = item.get('diffusion_type', '')
                    if diff_type in furniture_types:
                        all_furniture_sizes_x.append(item['size_x'])
                        all_furniture_sizes_z.append(item['size_z'])
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    print(f"Found {len(all_furniture_sizes_x)} furniture items")
else:
    print(f"Warning: JSON directory not found: {json_dir}")

# 3. 打印统计结果
print("\n" + "="*60)
print("=== ROOM STATISTICS (Real Meters) ===")
print("="*60)

if all_room_widths:
    room_widths = np.array(all_room_widths)
    room_depths = np.array(all_room_depths)

    print(f"Count:      {len(room_widths)} rooms")
    print(f"Mean room:  ({room_widths.mean():.3f}, {room_depths.mean():.3f}) meters")
    print(f"Median:     ({np.median(room_widths):.3f}, {np.median(room_depths):.3f}) meters")
    print(f"Std:        ({room_widths.std():.3f}, {room_depths.std():.3f}) meters")
    print(f"Min room:   ({room_widths.min():.3f}, {room_depths.min():.3f}) meters")
    print(f"Max room:   ({room_widths.max():.3f}, {room_depths.max():.3f}) meters")
    print(f"95th %ile:  ({np.percentile(room_widths, 95):.3f}, {np.percentile(room_depths, 95):.3f}) meters")
else:
    print("No room data found!")

print("\n" + "="*60)
print("=== FURNITURE SIZE STATISTICS (Real Meters) ===")
print("="*60)

if all_furniture_sizes_x:
    furn_x = np.array(all_furniture_sizes_x)
    furn_z = np.array(all_furniture_sizes_z)

    print(f"Count:      {len(furn_x)} furniture items")
    print(f"Mean size:  ({furn_x.mean():.3f}, {furn_z.mean():.3f}) meters")
    print(f"Median:     ({np.median(furn_x):.3f}, {np.median(furn_z):.3f}) meters")
    print(f"Std:        ({furn_x.std():.3f}, {furn_z.std():.3f}) meters")
    print(f"Min size:   ({furn_x.min():.3f}, {furn_z.min():.3f}) meters")
    print(f"Max size:   ({furn_x.max():.3f}, {furn_z.max():.3f}) meters")
    print(f"95th %ile:  ({np.percentile(furn_x, 95):.3f}, {np.percentile(furn_z, 95):.3f}) meters")
else:
    print("No furniture data found!")

# 4. 建议的全局归一化参数
print("\n" + "="*60)
print("=== RECOMMENDED GLOBAL NORMALIZATION ===")
print("="*60)

if all_room_widths and all_furniture_sizes_x:
    mean_room_w = np.mean(all_room_widths)
    mean_room_d = np.mean(all_room_depths)
    max_room_w = np.max(all_room_widths)
    max_room_d = np.max(all_room_depths)

    print(f"\nOption 1: Use mean room size")
    print(f"  GLOBAL_ROOM_WIDTH = {mean_room_w:.2f}")
    print(f"  GLOBAL_ROOM_DEPTH = {mean_room_d:.2f}")

    print(f"\nOption 2: Use max room size (safer)")
    print(f"  GLOBAL_ROOM_WIDTH = {max_room_w:.2f}")
    print(f"  GLOBAL_ROOM_DEPTH = {max_room_d:.2f}")

    print(f"\nOption 3: Use 95th percentile (balanced)")
    print(f"  GLOBAL_ROOM_WIDTH = {np.percentile(all_room_widths, 95):.2f}")
    print(f"  GLOBAL_ROOM_DEPTH = {np.percentile(all_room_depths, 95):.2f}")

    # 检查当前归一化后的分布
    print("\n" + "="*60)
    print("=== CURRENT NORMALIZATION (per-room) ===")
    print("="*60)

    # 读取已预处理的数据
    all_normalized_sizes = []
    count = 0
    for scene_dir in list(preprocessed_dir.iterdir())[:100]:
        if not scene_dir.is_dir():
            continue
        npz_path = scene_dir / 'boxes.npz'
        if npz_path.exists():
            data = np.load(npz_path)
            sizes = data['sizes']
            all_normalized_sizes.extend(sizes.tolist())
            count += 1

    if all_normalized_sizes:
        norm_sizes = np.array(all_normalized_sizes)
        print(f"Analyzed {count} preprocessed scenes")
        print(f"Current normalized size mean: ({norm_sizes[:, 0].mean():.3f}, {norm_sizes[:, 1].mean():.3f})")
        print(f"Current normalized size std:  ({norm_sizes[:, 0].std():.3f}, {norm_sizes[:, 1].std():.3f})")
        print(f"Current normalized size max:  ({norm_sizes[:, 0].max():.3f}, {norm_sizes[:, 1].max():.3f})")

        print("\nWith global normalization (mean room):")
        global_norm_x = furn_x / mean_room_w
        global_norm_z = furn_z / mean_room_d
        print(f"  Mean: ({global_norm_x.mean():.3f}, {global_norm_z.mean():.3f})")
        print(f"  Std:  ({global_norm_x.std():.3f}, {global_norm_z.std():.3f})")
        print(f"  Max:  ({global_norm_x.max():.3f}, {global_norm_z.max():.3f})")

print("\n" + "="*60)
print("Done!")
print("="*60)
