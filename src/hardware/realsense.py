import pyrealsense2 as rs
import numpy as np

class RealSenseCamera:
    """
    Interface for Intel RealSense D405 series camera
    Handles stream initialization, frame allignment, and raw data extraction
    """
    def __init__(self, width=640, height=480, fps=30):
        """
        Sets up the RealSense pipeline and configuration for depth and color streams
        
        Args: 
            width (int): Width of the video stream
            height (int): Height of the video stream
            fps (int): Frames per second for the video stream
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.align = rs.align(rs.stream.color)
        self.profile = None

    def start(self):
        """Start the camera hardware pipeline"""
        try:
            self.profile = self.pipeline.start(self.config)
            print("RealSense camera started successfully.")
        except Exception as e:
            print(f"Failed to start RealSense camera: {e}")
            raise e
        
    def get_frames(self, timeout_ms=5000):
        """Retrieves and aligns the latest color and depth frames from the camera

        Returns:
            tuple: (color_image as numpy array, depth_frame object) or (None, None) if retrieval fails
        """
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
        except Exception as e:
            print(f"Failed to get frames: {e}")
            return None, None
        
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("Failed to get valid frames.")
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame

    def stop(self):
        """Stops the camera pipeline and releases hardware resources"""
        self.pipeline.stop()
        print("RealSense camera stopped.")
