import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import RTDETR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DETR_WEIGHTS = 'weights/best.pt'

# smoothing storage
prev_positions = {}

def smooth(key, new, alpha=0.7):
    if key not in prev_positions:
        prev_positions[key] = new
        return new
    prev = prev_positions[key]
    smoothed = alpha * prev + (1 - alpha) * new
    prev_positions[key] = smoothed
    return smoothed


def compute_pca_orientation(points_3d):
    if len(points_3d) < 10:
        return None
    
    mean = np.mean(points_3d, axis=0)
    centered = points_3d - mean
    cov = np.cov(centered.T)
    
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    return principal_axis


def run_realsense_pipeline():
    print("Loading DETR model...")
    detr_model = RTDETR(DETR_WEIGHTS)

    print("Initializing RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    print("Running... Press 'q' to quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # DETECTION
            results = detr_model.predict(rgb_image, conf=0.8, verbose=False)

            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            for i in range(len(results[0].boxes)):
                box = results[0].boxes.xyxy[i].cpu().numpy()
                x_min, y_min, x_max, y_max = map(int, box)

                class_id = int(results[0].boxes.cls[i].cpu().numpy())
                confidence = float(results[0].boxes.conf[i].cpu().numpy())
                class_name = detr_model.names[class_id]

                # clamp ROI
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(639, x_max)
                y_max = min(479, y_max)

                roi = depth_image[y_min:y_max, x_min:x_max]
                valid = roi[roi > 0]

                if len(valid) < 20:
                    continue

                depth = np.median(valid) * 0.001  # mm → meters

                cx = int((x_min + x_max) / 2)
                cy = int((y_min + y_max) / 2)

                X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)

                # smoothing
                key = f"{class_id}_{i}"
                X, Y, Z = smooth(key, np.array([X, Y, Z]))

                # PCA orientation (sample points)
                sample_points = []
                step = max(1, (x_max - x_min)//20)

                for y in range(y_min, y_max, step):
                    for x in range(x_min, x_max, step):
                        d = depth_image[y, x]
                        if d > 0:
                            d_m = d * 0.001
                            pt = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d_m)
                            sample_points.append(pt)

                orientation = compute_pca_orientation(np.array(sample_points))

                # VISUALIZATION
                cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                label = f"{class_name} {confidence:.2f}"
                cv2.putText(color_image, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                cv2.drawMarker(color_image, (cx, cy), (0, 0, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

                coord_text = f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}m"
                cv2.putText(color_image, coord_text, (cx+10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # draw orientation vector
                if orientation is not None:
                    end_point = (int(cx + orientation[0]*50),
                                 int(cy + orientation[1]*50))
                    cv2.arrowedLine(color_image, (cx, cy), end_point, (0, 255, 0), 2)

            cv2.imshow("Pipeline (No SAM)", color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realsense_pipeline()