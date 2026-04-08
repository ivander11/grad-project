import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

def is_blurry(image, threshold=50):
    """Calculates the variance of the Laplacian to measure focus."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def record_dataset_frames(output_dir="data/marker_glue_meds", fps_target=1, blur_threshold=50):
    """Records frames from a RealSense camera, saving only those that are not blurry."""
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)
    
    # Highly recommended: Lock exposure to prevent motion blur indoors
    # color_sensor = profile.get_device().query_sensors()[1]
    # color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    # color_sensor.set_option(rs.option.exposure, 150)
    
    print(f"Saving lossless frames to {output_dir} at max {fps_target} FPS.")
    print("Move the camera or objects. Press 'q' to stop recording.")
    
    frame_interval = 1.0 / fps_target
    last_check_time = time.time()
    saved_count = 0
    dropped_count = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('Recording Dataset (Press Q to quit)', color_image)

            current_time = time.time()
            
            # Check if it is time to sample a frame
            if current_time - last_check_time >= frame_interval:
                last_check_time = current_time # Reset timer
                
                # Apply the mathematical blur filter
                if not is_blurry(color_image, threshold=blur_threshold):
                    filename = os.path.join(output_dir, f"frame_{saved_count:05d}.png")
                    cv2.imwrite(filename, color_image)
                    saved_count += 1
                    print(f"Saved: {saved_count} | Dropped (Blurry): {dropped_count}", end='\r')
                else:
                    dropped_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\nExecution complete. Saved {saved_count} sharp frames. Dropped {dropped_count} frames.")

if __name__ == "__main__":
    record_dataset_frames()