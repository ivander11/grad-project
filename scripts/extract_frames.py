import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, fps_target=2):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps_original = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate exact frame step
    frame_step = max(1, int(round(fps_original / fps_target)))
    
    # Get base name for unique file naming
    video_name = Path(video_path).stem
    
    saved_count = 0
    current_frame = 0
    
    while current_frame < total_frames:
        # Jump directly to the target frame, skipping intermediate decoding
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Use .png for lossless quality, crucial for SAM segmentation masks
        filename = os.path.join(output_dir, f"{video_name}_frame_{saved_count:05d}.png")
        cv2.imwrite(filename, frame)
        
        saved_count += 1
        current_frame += frame_step
        
    cap.release()
    print(f"Extracted {saved_count} frames from {video_name} to {output_dir}")

# Usage
# extract_frames("video1.mp4", "dataset/images", fps_target=2)