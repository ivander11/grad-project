import os
import glob
from pathlib import Path

# 1. SETUP: Update paths
DATASET_ROOT = r"C:\Ivander\rl_grasping_project\dataset\rt_detr_dataset\yolo_format"
DIRS = ["train", "valid", "test"]
THRESHOLD = 0.02  
EXTENSIONS = [".png", ".jpg", ".jpeg"]

unique_signatures = []
deleted_count = 0

def get_scene_signature(label_path):
    """Extracts and sorts coordinates. Returns empty list for background images."""
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        data = [line.strip().split() for line in f.readlines() if line.strip()]
        # Filter for valid YOLO lines and sort by x-center
        coords = sorted([(float(p[1]), float(p[2])) for p in data if len(p) >= 5], key=lambda x: x[0])
    return coords

def is_duplicate(new_sig):
    """Handles math for empty and populated signatures."""
    for seen_sig in unique_signatures:
        # If box counts differ, they are not duplicates
        if len(new_sig) != len(seen_sig):
            continue
        
        # Handle Case: Both are background images (0 boxes)
        if len(new_sig) == 0:
            return True
        
        # Calculate Euclidean distance for populated images
        total_dist = sum([((n[0]-s[0])**2 + (n[1]-s[1])**2)**0.5 for n, s in zip(new_sig, seen_sig)])
        avg_dist = total_dist / len(new_sig)
        
        if avg_dist < THRESHOLD:
            return True
    return False

print("Starting Global Deduplication...")

for folder in DIRS:
    label_path = os.path.join(DATASET_ROOT, folder, "labels")
    image_path = os.path.join(DATASET_ROOT, folder, "images")
    
    if not os.path.exists(label_path):
        print(f"Skipping {folder}: Path not found.")
        continue

    files = glob.glob(os.path.join(label_path, "*.txt"))
    
    for f_path in files:
        if os.path.basename(f_path) == "classes.txt":
            continue
            
        current_sig = get_scene_signature(f_path)
        
        if is_duplicate(current_sig):
            base_name = Path(f_path).stem
            os.remove(f_path)
            for ext in EXTENSIONS:
                img_f = os.path.join(image_path, base_name + ext)
                if os.path.exists(img_f):
                    os.remove(img_f)
            deleted_count += 1
        else:
            unique_signatures.append(current_sig)

print(f"\nCleanup Complete.")
print(f"Unique scenes identified: {len(unique_signatures)}")
print(f"Total duplicates removed: {deleted_count}")