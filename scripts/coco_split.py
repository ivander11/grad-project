import json
import os
import shutil
import random

def execute_coco_split(source_dir, output_dir, train_ratio=0.70, valid_ratio=0.20):
    json_path = os.path.join(source_dir, '_annotations.coco.json')
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data.get('categories', [])

    # Randomize to prevent sequential video frame bias
    random.seed(42)
    random.shuffle(images)

    total_images = len(images)
    train_index = int(total_images * train_ratio)
    valid_index = train_index + int(total_images * valid_ratio)

    splits = {
        'train': images[:train_index],
        'valid': images[train_index:valid_index],
        'test': images[valid_index:]
    }

    # Map annotations to image IDs for O(1) lookup speed
    ann_map = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in ann_map:
            ann_map[img_id] = []
        ann_map[img_id].append(ann)

    print(f"Total images found: {total_images}")
    print("Executing split and generating matrices...")

    for split_name, split_images in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        split_annotations = []
        
        for img in split_images:
            # Execute physical file copy
            src_file = os.path.join(source_dir, img['file_name'])
            dst_file = os.path.join(split_dir, img['file_name'])
            
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
            else:
                print(f"Warning: Physical file missing - {img['file_name']}")

            # Extract corresponding bounding box coordinates
            img_id = img['id']
            if img_id in ann_map:
                split_annotations.extend(ann_map[img_id])

        # Compile new JSON structure
        split_coco_json = {
            'images': split_images,
            'annotations': split_annotations,
            'categories': categories
        }

        # Write split JSON to disk
        output_json_path = os.path.join(split_dir, '_annotations.coco.json')
        with open(output_json_path, 'w') as f:
            json.dump(split_coco_json, f)

        print(f"Compiled {split_name}: {len(split_images)} images")

# --- CONFIGURATION ---
SOURCE_DIR = r'C:\Ivander\rl_grasping_project\dataset\RT_DETR Datasets.coco-mmdetection\train'
OUTPUT_DIR = r'C:\Ivander\rl_grasping_project\dataset\rt_detr_dataset'

execute_coco_split(SOURCE_DIR, OUTPUT_DIR)