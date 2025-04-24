import time
from threading import Thread
from collections import deque
import cv2

from script.video_stream import VideoStream
from script.face_mask_detector import FaceMaskDetector

class StreamlitDetector:
    """
    Wrapper class to integrate FaceMaskDetector with Streamlit
    """
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.running = False
        self.video_stream = None
        self.detector = None
        self.frame = None
        self.processing_times = deque(maxlen=30)
        self.initialize_detector()
        
    def initialize_detector(self):
        try:
            self.detector = FaceMaskDetector()
            return True
        except Exception as e:
            print(f"Error initializing detector: {e}")
            return False
            
    def start(self):
        if not self.running:
            try:
                self.video_stream = VideoStream(src=0).start()
                # Wait for camera to initialize
                time.sleep(2.0)
                
                # Check if video stream is working
                test_frame = self.video_stream.read()
                if test_frame is None:
                    print("Failed to get frame from camera")
                    return False
                
                self.running = True
                self.processing_thread = Thread(target=self.process_frames)
                self.processing_thread.daemon = True
                self.processing_thread.start()
                return True
            except Exception as e:
                print(f"Error starting video stream: {e}")
                return False
    
    def process_image_debug(self, image):
        """
        Process a single image with debugging information
        Returns a list of tuples containing step-by-step processing results:
        [(title, image, description), ...]
        """
        try:
            if image is None:
                return [("Error", None, "Invalid image provided")]
            
            # Resize image for consistent processing
            image = cv2.resize(image, (self.width, self.height))
            
            # Process image with debugging
            _, _, debug_steps = self.detector.process_frame(image, debug=True)
            
            return debug_steps
            
        except Exception as e:
            print(f"Error in debug image processing: {e}")
            return [("Error", None, f"Error processing image: {str(e)}")]
            
    def process_frames(self):
        while self.running:
            try:
                if self.video_stream and not self.video_stream.stopped:
                    frame = self.video_stream.read()
                    if frame is not None:
                        # Resize frame
                        frame = cv2.resize(frame, (self.width, self.height))
                        
                        # Process frame and measure time
                        process_start = time.time()
                        output, num_faces = self.detector.process_frame(frame)
                        process_end = time.time()
                        
                        # Calculate processing time
                        process_time = process_end - process_start
                        self.processing_times.append(process_time)
                        
                        # Calculate average processing time
                        avg_process_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
                        
                        # Calculate FPS
                        fps = self.video_stream.get_fps()
                        
                        # Display performance metrics
                        info_text = f"FPS: {fps:.2f}, Faces: {num_faces}, Process: {avg_process_time*1000:.1f}ms"
                        cv2.putText(output, info_text, (10, 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Update the frame
                        self.frame = output
            except Exception as e:
                print(f"Error in processing frame: {e}")
            
            # Small sleep to prevent high CPU usage
            time.sleep(0.01)
            
    def get_frame(self):
        return self.frame
            
    def stop(self):
        self.running = False
        if self.video_stream:
            self.video_stream.stop()
            self.video_stream = None