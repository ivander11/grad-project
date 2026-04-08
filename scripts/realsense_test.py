import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import RTDETR
from mobile_sam import sam_model_registry, SamPredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DETR_WEIGHTS = 'weights/best.pt'
SAM_WEIGHTS = 'weights/mobile_sam.pt'

def run_realsense_pipeline():
    print("Loading models into VRAM...")
    detr_model = RTDETR(DETR_WEIGHTS)
    
    mobile_sam = sam_model_registry["vit_t"](checkpoint=SAM_WEIGHTS)
    mobile_sam.to(device=DEVICE)
    mobile_sam.eval()
    sam_predictor = SamPredictor(mobile_sam)

    print("Initializing Intel RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"CRITICAL ERROR: RealSense not detected.\n{e}")
        return

    align_to = rs.stream.color
    align = rs.align(align_to)

    print("RealSense active. Press 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # 1. Detection
            results = detr_model.predict(rgb_image, conf=0.85, verbose=False)
            
            if len(results[0].boxes) > 0:
                # Grab the highest confidence detection
                box = results[0].boxes.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = map(int, box)
                
                # Extract classification data
                class_id = int(results[0].boxes.cls[0].cpu().numpy())
                confidence = float(results[0].boxes.conf[0].cpu().numpy())
                class_name = detr_model.names[class_id]

                # 2. Segmentation
                sam_predictor.set_image(rgb_image)
                masks, _, _ = sam_predictor.predict(box=box, multimask_output=False)
                target_mask = masks[0]

                # 3. Centroid & Depth Extraction
                y_pixels, x_pixels = np.where(target_mask == True)
                if len(x_pixels) > 0:
                    cx = int(np.mean(x_pixels))
                    cy = int(np.mean(y_pixels))

                    depth_meters = depth_frame.get_distance(cx, cy)
                    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                    spatial_coords = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth_meters)
                    x_m, y_m, z_m = spatial_coords[0], spatial_coords[1], spatial_coords[2]

                    # --- VISUALIZATION ---
                    # Draw Bounding Box
                    cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    
                    # Draw Label and Confidence
                    label_text = f"{class_name} {confidence:.2f}"
                    cv2.putText(color_image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Draw Mask
                    green_mask = np.zeros_like(color_image, dtype=np.uint8)
                    green_mask[target_mask] = [0, 255, 0] 
                    color_image = cv2.addWeighted(color_image, 1.0, green_mask, 0.4, 0)

                    # Draw Crosshair
                    cv2.drawMarker(color_image, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                    
                    # Draw Coordinates
                    coord_text = f"X:{x_m:.3f} Y:{y_m:.3f} Z:{z_m:.3f}m"
                    cv2.putText(color_image, coord_text, (cx + 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow('RealSense RL Vision Pipeline', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realsense_pipeline()