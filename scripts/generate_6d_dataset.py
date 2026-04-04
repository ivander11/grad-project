import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
from scipy.spatial.transform import Rotation

# --- Configuration ---
MARKER_SIZE = 0.02  
MARKER_ID = 0      
CLASS_ID = 0        
DATASET_DIR = "custom_6d_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

os.makedirs(IMAGES_DIR, exist_ok=True)

# Define the 3D coordinates of the marker corners in the marker's local coordinate system
# Ordering matches the OpenCV ArUco detection output (top-left, top-right, bottom-right, bottom-left)
obj_points = np.array([
    [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
], dtype=np.float32)

# --- RealSense Initialization ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Get camera intrinsics
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)

# --- ArUco Initialization (OpenCV 4.7+ API) ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

annotations = []
frame_count = 0
recording = False

print("System Ready.")
print("Press 'R' to toggle recording.")
print("Press 'Q' to quit and save the dataset.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)
        display_img = img.copy()

        if ids is not None and MARKER_ID in ids:
            # Find the specific marker index
            idx = np.where(ids == MARKER_ID)[0][0]
            marker_corners = corners[idx][0]

            # 1. Bounding Box Extraction (2D)
            x_min = float(np.min(marker_corners[:, 0]))
            y_min = float(np.min(marker_corners[:, 1]))
            x_max = float(np.max(marker_corners[:, 0]))
            y_max = float(np.max(marker_corners[:, 1]))

            # 2. 6D Pose Estimation (3D Translation + Rotation)
            success, rvec, tvec = cv2.solvePnP(
                obj_points, marker_corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if success:
                # Format Translation Vector [X, Y, Z] in meters
                centers_3d = [float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])]

                # Format Rotation Vector to Quaternion [q_w, q_x, q_y, q_z]
                rot = Rotation.from_rotvec(rvec.flatten())
                quat = rot.as_quat() # Scipy outputs [x, y, z, w]
                orientations = [float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])]

                # Draw visualization
                cv2.aruco.drawDetectedMarkers(display_img, corners, ids)
                cv2.drawFrameAxes(display_img, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

                if recording:
                    # Save Image
                    filename = f"{frame_count:06d}.jpg"
                    filepath = os.path.join(IMAGES_DIR, filename)
                    cv2.imwrite(filepath, img)

                    # Append to JSON structure
                    annotations.append({
                        "image_id": frame_count,
                        "file_name": filename,
                        "class_id": CLASS_ID,
                        "boxes": [x_min, y_min, x_max, y_max],
                        "centers": centers_3d,
                        "orientations": orientations
                    })
                    frame_count += 1
                    cv2.putText(display_img, f"RECORDING: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            if recording:
                cv2.putText(display_img, f"RECORDING: {frame_count} (NO MARKER)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        cv2.imshow("6D Dataset Generator", display_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recording = not recording
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    # Save the dataset dictionary
    dataset_dict = {"annotations": annotations}
    json_path = os.path.join(DATASET_DIR, "annotations.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_dict, f, indent=4)
    
    print(f"Dataset saved. Total frames: {frame_count}")
    print(f"Output directory: {os.path.abspath(DATASET_DIR)}")