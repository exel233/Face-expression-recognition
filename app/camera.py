import cv2
import numpy as np
from threading import Thread
from collections import deque
from typing import Optional, Tuple
import time

class CameraCapture:
    """Handles camera capture and frame processing"""
    
    def __init__(self, camera_index: int = 0, frame_rate: int = 30):
        self.cap = None
        self.camera_index = camera_index
        self.frame_rate = frame_rate
        self.frame = None
        self.ret = False
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.error_count = 0
        self.max_errors = 10
        
    def start(self):
        """Start capturing frames in a background thread"""
        # Try different camera indices if the default fails
        for idx in [self.camera_index, 0, 1, 2]:
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened():
                print(f"Successfully opened camera {idx}")
                break
            self.cap.release()
        
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("No camera found or cannot be opened. Please check your webcam connection.")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
        
        self.running = True
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        """Background thread loop for capturing frames"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            try:
                self.ret, self.frame = self.cap.read()
                
                if not self.ret or self.frame is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Camera capture failed {consecutive_errors} times, attempting to reconnect...")
                        self._reconnect_camera()
                        consecutive_errors = 0
                    time.sleep(0.1)
                    continue
                
                consecutive_errors = 0
                
                # Calculate FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_time = current_time
                
            except Exception as e:
                print(f"Error in camera capture loop: {e}")
                consecutive_errors += 1
                time.sleep(0.1)
            
            time.sleep(0.01)  # Limit to ~100 FPS
    
    def _reconnect_camera(self):
        """Attempt to reconnect to the camera"""
        try:
            if self.cap:
                self.cap.release()
            
            # Try to reopen camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
                print("Successfully reconnected to camera")
            else:
                print("Failed to reconnect to camera")
        except Exception as e:
            print(f"Error reconnecting camera: {e}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame"""
        return self.frame.copy() if self.frame is not None else None
    
    def is_active(self) -> bool:
        """Check if camera is actively capturing"""
        return self.running and self.ret
    
    def stop(self):
        """Stop capturing frames"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.cap.release()
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()


class FrameProcessor:
    """Processes frames for display and processing"""
    
    @staticmethod
    def add_text_overlay(image: np.ndarray, text: str, position: Tuple[int, int],
                        color: Tuple[int, int, int] = (0, 255, 0),
                        font_size: float = 1.0, thickness: int = 2) -> np.ndarray:
        """Add text overlay to image"""
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   font_size, color, thickness, cv2.LINE_AA)
        return image
    
    @staticmethod
    def add_fps(image: np.ndarray, fps: int) -> np.ndarray:
        """Add FPS counter to image"""
        cv2.putText(image, f"FPS: {fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return image
    
    @staticmethod
    def draw_box(image: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int,
                color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Draw rectangle on image"""
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        return image
    
    @staticmethod
    def resize_frame(image: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
        """Resize frame to specified dimensions"""
        return cv2.resize(image, (width, height))
    
    @staticmethod
    def flip_frame(image: np.ndarray, horizontal: bool = True) -> np.ndarray:
        """Flip frame horizontally or vertically"""
        if horizontal:
            return cv2.flip(image, 1)
        return cv2.flip(image, 0)


class FrameEncoder:
    """Converts OpenCV frames to JPEG for streaming"""
    
    @staticmethod
    def encode_frame(frame: np.ndarray, quality: int = 80) -> bytes:
        """Encode frame to JPEG bytes
        
        Args:
            frame: OpenCV image frame
            quality: JPEG quality (0-100)
        
        Returns:
            JPEG encoded bytes
        """
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ret:
            return None
        return buffer.tobytes()
