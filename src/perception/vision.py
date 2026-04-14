import torch
import numpy as np
import cv2
from ultralytics import RTDETR
from mobile_sam import sam_model_registry, SamPredictor
import pyrealsense2 as rs

class VisionPipeline:
    """
    AI Perception engine for robotic grasping
    Integrates RT-DETR (Detection), Mobile SAM (Segmentation), and 3D Deprojection
    """
    def __init__(self, detr_weights="../models/best.pt", sam_weights="../models/mobile_sam.pt", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("Loading RT-DETR...")
        self.detr_model = RTDETR(detr_weights)
        
        print("Loading Mobile SAM...")
        mobile_sam = sam_model_registry["vit_t"](checkpoint=sam_weights)
        mobile_sam.to(device=self.device)
        mobile_sam.eval()
        self.sam_predictor = SamPredictor(mobile_sam)

    def get_class_color(self, class_id: int):
        """Generates a consistent BGR color based on the object class ID"""
        hue = (class_id * 37) % 180
        hsv_color = np.uint8([[[hue, 220, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        return int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])

    def detect(self, rgb_image, conf_threshold=0.5):
        """
        Executes object detection
        
        Returns:
            ultralytics.engine.results.Results: Detection data including bounding boxes
        """
        return self.detr_model.predict(rgb_image, conf=conf_threshold, verbose=False)[0]

    def segment(self, rgb_image, box):
        """"Generates a high-precision mask for a specific bounding box
        
        Args:
            rgb_image (np.array): RGB frame from camera
            box (list): [x_min, y_min, x_max, y_max] from detector
            
        Returns:
            np.array: Boolean mask where True represents the object pixels
        """
        self.sam_predictor.set_image(rgb_image)
        x_min, y_min, x_max, y_max = map(int, box)
        box_cx = int((x_min + x_max) / 2)
        box_cy = int((y_min + y_max) / 2)
        
        input_point = np.array([[box_cx, box_cy]])
        input_label = np.array([1])

        masks, _, _ = self.sam_predictor.predict(
            box=box, 
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        return masks[0]

    def extract_3d_coordinates(self, depth_frame, mask):
        """
        Converts 2D mask information into real-world 3D coordinates (meters)
        
        Args:
            depth_frame (rs.frame): The depth data from RealSense
            mask (np.array): The boolean segmentation mask
            
        Returns:
            tuple: ([x, y, z] in meters, centroid_x, centroid_y)
        """
        y_pixels, x_pixels = np.where(mask == True)
        if len(x_pixels) == 0:
            return None, None, None

        cx, cy = int(np.mean(x_pixels)), int(np.mean(y_pixels))
        
        depth_data = np.asanyarray(depth_frame.get_data())
        depth_region = depth_data[max(0, cy-2):min(480, cy+3), max(0, cx-2):min(640, cx+3)]
        valid_depths = depth_region[depth_region > 0]
        
        if len(valid_depths) == 0:
            return None, cx, cy
            
        depth_meters = np.median(valid_depths) * depth_frame.get_units()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        spatial_coords = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth_meters)
        
        return spatial_coords, cx, cy