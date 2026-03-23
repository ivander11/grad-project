import cv2
import torch
import numpy as np
from ultralytics import RTDETR
from mobile_sam import sam_model_registry, SamPredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DETR_WEIGHTS = 'weights/best.pt'
SAM_WEIGHTS = 'weights/mobile_sam.pt'

def run_live_inference():
    print("Loading models into VRAM...")
    detr_model = RTDETR(DETR_WEIGHTS)
    
    mobile_sam = sam_model_registry["vit_t"](checkpoint=SAM_WEIGHTS)
    mobile_sam.to(device=DEVICE)
    mobile_sam.eval()
    sam_predictor = SamPredictor(mobile_sam)
    
    # Initialize Webcam (0 is usually the built-in laptop camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting live feed. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV uses BGR. Models require RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Run RT-DETR Detection
        results = detr_model.predict(rgb_frame, conf=0.85, verbose=False)
        
        if len(results[0].boxes) > 0:
            # Loop through EVERY detected object
            for i in range(len(results[0].boxes)):
                box = results[0].boxes.xyxy[i].cpu().numpy()
                x_min, y_min, x_max, y_max = map(int, box)
                    
                # Extract classification data for current object
                class_id = int(results[0].boxes.cls[i].cpu().numpy())
                confidence = float(results[0].boxes.conf[i].cpu().numpy())
                class_name = detr_model.names[class_id]

            # 2. Run MobileSAM Segmentation
            sam_predictor.set_image(rgb_frame)
            masks, _, _ = sam_predictor.predict(box=box, multimask_output=False)
            target_mask = masks[0]

            # 3. Calculate Centroid
            y_pixels, x_pixels = np.where(target_mask == True)
            if len(x_pixels) > 0:
                cx = int(np.mean(x_pixels))
                cy = int(np.mean(y_pixels))

                # --- VISUALIZATION OVERLAYS ---
                
                # Draw Bounding Box (Blue)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                
                # Draw Segmentation Mask (Green Overlay)
                green_mask = np.zeros_like(frame, dtype=np.uint8)
                green_mask[target_mask] = [0, 255, 0] 
                frame = cv2.addWeighted(frame, 1.0, green_mask, 0.4, 0)

                # Draw Centroid Crosshair (Red)
                cv2.drawMarker(frame, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Display the output
        cv2.imshow('RT-DETR + MobileSAM Live Pipeline', frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_inference()