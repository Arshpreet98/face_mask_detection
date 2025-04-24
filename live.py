import cv2
import numpy as np
import time
from threading import Thread
import os
from collections import deque
import json

class VideoStream:
    """
    Class for efficient video streaming using threading
    """
    def __init__(self, src=0, name="VideoStream"):
        self.stream = cv2.VideoCapture(src)
        self.name = name
        self.stopped = False
        self.frame = None
        self.grabbed = False
        (self.grabbed, self.frame) = self.stream.read()
        self.fps_deque = deque(maxlen=30)  # Store last 30 frame timestamps for FPS calculation

    def start(self):
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
                continue
            (self.grabbed, self.frame) = self.stream.read()
            self.fps_deque.append(time.time())

    def read(self):
        return self.frame

    def get_fps(self):
        if len(self.fps_deque) <= 1:
            return 0
        return len(self.fps_deque) / (self.fps_deque[-1] - self.fps_deque[0])

    def stop(self):
        self.stopped = True
        self.stream.release()


class FaceMaskDetector:
    """
    Class for face mask detection
    """
    # Color codes for different mask statuses (BGR format)
    COLORS = {
        "with_mask": (0, 255, 0),      # Green
        "without_mask": (0, 0, 255),    # Red
        "incorrect_mask": (0, 255, 255) # Yellow
    }
    
    # Status labels
    LABELS = {
        0: "With Mask",
        1: "Without Mask", 
        2: "Incorrect Mask"
    }

    def __init__(self, 
                 mask_model_path="model/final_mask_detection_model.h5", 
                 face_model_path="model/res10_300x300_ssd_iter_140000.caffemodel",
                 prototxt_path="model/deploy.prototxt",
                 threshold_path="model/optimal_thresholds.json",
                 confidence_threshold=0.5):
        
        self.confidence_threshold = confidence_threshold
        
        # Load thresholds
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
        self.class_names = threshold_data["class_names"]
        self.thresholds = threshold_data["thresholds"]
        
        # Map class names to indices
        self.class_to_index = {name: i for i, name in enumerate(self.class_names)}
        
        # Update LABELS based on class names
        self.LABELS = {
            i: name.capitalize() for i, name in enumerate(self.class_names)
        }
        
        print(f"[INFO] Loaded custom thresholds: {self.thresholds}")
        print(f"[INFO] Classes: {self.class_names}")

        # Check if model paths exist
        for path in [mask_model_path, face_model_path, prototxt_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load the face detector model
        print("[INFO] Loading face detector model...")
        self.face_net = cv2.dnn.readNet(prototxt_path, face_model_path)
        
        # Load the mask detector model
        print("[INFO] Loading mask detector model...")
        try:
            # Try to load with TensorFlow
            try:
                import tensorflow as tf
                self.mask_net = tf.keras.models.load_model(mask_model_path)
                self.framework = "tensorflow"
            except Exception as e:
                print("Error:", e)
                raise Exception(f"Failed to load mask detection model with TensorFlow: {e}")
        except Exception as e:
            raise Exception(f"Failed to load mask detection model: {e}")
        
        print("[INFO] Models loaded successfully.")
        print(f"[INFO] Using {self.framework} framework for mask detection.")

    def detect_faces(self, frame):
        """
        Detect faces in the frame using OpenCV DNN
        """
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # Pass the blob through the face detection network
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        
        # Process detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter weak detections
            if confidence > self.confidence_threshold:
                # Calculate bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding box dimensions are within frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Extract face ROI
                face = frame[startY:endY, startX:endX]
                
                # Skip if face ROI is empty
                if face.size == 0:
                    continue
                
                faces.append({
                    "box": (startX, startY, endX, endY),
                    "roi": face,
                    "confidence": confidence
                })
                
        return faces

    def predict_mask(self, face_roi):
        """
        Predict mask status for a face ROI
        """
        try:
            # Preprocess face for mask detection
            face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (300, 300))
            face = face.astype("float") / 255.0
            face = np.expand_dims(face, axis=0)
            
            if self.framework == "tensorflow":
                # Make prediction with TensorFlow model
                prediction = self.mask_net.predict(face, verbose=0)[0]

                # Apply thresholds
                max_index = -1
                max_confidence = -1
                
                for i, (class_name, threshold) in enumerate(zip(self.class_names, self.thresholds)):
                    if prediction[i] > threshold and prediction[i] > max_confidence:
                        max_confidence = prediction[i]
                        max_index = i
                
                # If no class meets threshold, choose the highest probability
                if max_index == -1:
                    max_index = np.argmax(prediction)
                
                label = self.LABELS[max_index]
                confidence = float(prediction[max_index])
                
            else:  # OpenCV DNN
                # Prepare blob for OpenCV model
                blob = cv2.dnn.blobFromImage(face_roi, 1.0, (224, 224), (104.0, 177.0, 123.0))
                self.mask_net.setInput(blob)
                prediction = self.mask_net.forward()
                max_index = np.argmax(prediction[0])
                label = self.LABELS[max_index]
                confidence = float(prediction[0][max_index])
            
            # Map the index to color
            if max_index == 0:  # With mask
                color = self.COLORS["with_mask"]
            elif max_index == 1:  # Without mask
                color = self.COLORS["without_mask"]
            else:  # Incorrect mask
                color = self.COLORS["incorrect_mask"]
                
            return label, color, confidence
            
        except Exception as e:
            print(f"Error in mask prediction: {e}")
            # Return default values in case of error
            return "Error", (0, 0, 0), 0.0

    def process_frame(self, frame):
        """
        Process a frame: detect faces, predict mask status, and annotate
        """
        if frame is None:
            return None, 0
            
        # Create a copy to draw on
        output = frame.copy()
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Process each face
        for face_data in faces:
            (startX, startY, endX, endY) = face_data["box"]
            face_roi = face_data["roi"]
            
            # Predict mask status
            label, color, confidence = self.predict_mask(face_roi)
            
            # Draw bounding box
            cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
            
            # Create label text with confidence
            text = f"{label}: {confidence:.2f}"
            
            # Determine text position
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            # Add text with black background for better visibility
            cv2.rectangle(output, (startX, y-20), (startX + len(text)*11, y), color, -1)
            cv2.putText(output, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return output, len(faces)


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