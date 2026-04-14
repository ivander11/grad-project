import sys
import os
import cv2
import numpy as np

# Add the project root to the Python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hardware.realsense import RealSenseCamera
from src.perception.vision import VisionPipeline

def main():
    camera = RealSenseCamera()
    vision = VisionPipeline(detr_weights="../models/best.pt", sam_weights="../models/mobile_sam.pt")
    
    camera.start()
    print("Pipeline active. Press 'q' to quit.")

    try:
        while True:
            color_image, depth_frame = camera.get_frames()
            if color_image is None:
                continue

            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = vision.detect(rgb_image)

            if len(results.boxes) > 0:
                for i in range(len(results.boxes)):
                    box = results.boxes.xyxy[i].cpu().numpy()
                    x_min, y_min, x_max, y_max = map(int, box)
                    
                    class_id = int(results.boxes.cls[i].cpu().numpy())
                    confidence = float(results.boxes.conf[i].cpu().numpy())
                    class_name = vision.detr_model.names[class_id]
                    class_color = vision.get_class_color(class_id)

                    mask = vision.segment(rgb_image, box)
                    spatial_coords, cx, cy = vision.extract_3d_coordinates(depth_frame, mask)

                    # Visualization
                    cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), class_color, 2)
                    cv2.putText(color_image, f"{class_name} {confidence:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 2)
                    
                    class_mask = np.zeros_like(color_image, dtype=np.uint8)
                    class_mask[mask] = class_color
                    color_image = cv2.addWeighted(color_image, 1.0, class_mask, 0.4, 0)

                    if spatial_coords:
                        x_m, y_m, z_m = spatial_coords
                        cv2.drawMarker(color_image, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                        cv2.putText(color_image, f"X:{x_m:.3f} Y:{y_m:.3f} Z:{z_m:.3f}m", (cx + 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow('Modular Vision Test', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()