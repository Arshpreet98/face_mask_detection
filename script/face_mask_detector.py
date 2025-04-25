import cv2
import numpy as np
import os
import json

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
                 mask_model_path="script/model/final_mask_detection_model.h5", 
                 face_model_path="script/model/res10_300x300_ssd_iter_140000.caffemodel",
                 prototxt_path="script/model/deploy.prototxt",
                 threshold_path="script/model/optimal_thresholds.json",
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

    def detect_faces(self, frame, debug=False):
        """
        Detect faces in the frame using OpenCV DNN
        
        Returns:
            - faces: list of detected faces
            - debug_output: dict with debug info (if debug=True)
        """
        debug_output = {}
        (h, w) = frame.shape[:2]
        
        # Create a blob from the image - explain this step
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        if debug:
            # Visualize blob creation process
            blob_vis = []
            for i in range(3):  # For each channel (BGR)
                # Extract and denormalize the channel
                channel = blob[0, i, :, :]
                channel = (channel * 255).astype('uint8')
                blob_vis.append(channel)
            
            # Stack channels for visualization
            blob_visualization = cv2.merge(blob_vis)
            debug_output['blob_creation'] = blob_visualization
            debug_output['blob_description'] = (
                "Image is converted to a 'blob' - a 4D array (1x3x300x300) normalized to [0,1] range. "
                "The blob is mean-subtracted with values (104.0, 177.0, 123.0) for BGR channels respectively. "
                "This standardization helps the neural network process the image consistently."
            )
            
            # Store the resized image for debugging
            resized_frame = cv2.resize(frame, (300, 300))
            debug_output['resized_frame'] = resized_frame.copy()
            debug_output['resize_description'] = (
                "The image is resized to 300x300 pixels, which matches the input size "
                "expected by the face detection model (SSD with 300x300 input)."
            )
        
        # Pass the blob through the face detection network
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        if debug:
            debug_output['detections_description'] = (
                "The blob is passed through the SSD (Single Shot MultiBox Detector) network. "
                "The network outputs a 4D array where each detection contains: "
                "[batch_id, class_id, confidence, left, top, right, bottom]. "
                f"Found {detections.shape[2]} potential face detections."
            )
        
        faces = []
        
        # Create a copy of the frame for drawing bounding boxes
        if debug:
            detection_visualization = frame.copy()
        
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
                
                # Draw bounding box in the debug visualization
                if debug:
                    cv2.rectangle(detection_visualization, (startX, startY), (endX, endY), (255, 0, 0), 2)
                    # Add confidence text
                    cv2.putText(detection_visualization, f"Face: {confidence:.2f}", 
                                (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (255, 0, 0), 2)
                
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
        
        if debug:
            debug_output['detection_visualization'] = detection_visualization
            debug_output['num_faces'] = len(faces)
            debug_output['face_extraction_description'] = (
                f"After filtering detections by confidence (> {self.confidence_threshold}), "
                f"{len(faces)} valid face(s) were found. Each face region is extracted "
                "as a Region of Interest (ROI) for further processing."
            )
        
        if debug:
            return faces, debug_output
        else:
            return faces

    def predict_mask(self, face_roi, debug=False):
        """
        Predict mask status for a face ROI
        
        Returns:
            - label, color, confidence: mask prediction results
            - debug_output: dict with debug info (if debug=True)
        """
        debug_output = {}
        
        try:
            # Preprocess face for mask detection
            # Convert to RGB for the model
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            if debug:
                debug_output['face_rgb'] = face_rgb.copy()
                debug_output['rgb_description'] = (
                    "The face region is converted from BGR (OpenCV's default color format) "
                    "to RGB format because most deep learning models expect RGB input."
                )
            
            # Resize to the expected input size
            face_resized = cv2.resize(face_rgb, (300, 300))
            
            if debug:
                debug_output['face_resized'] = face_resized.copy()
                debug_output['resize_description'] = (
                    "The face is resized to 300x300 pixels to match the input dimensions "
                    "expected by the mask detection model. This ensures consistent processing "
                    "regardless of the original face size."
                )
            
            # Normalize pixel values
            face_normalized = face_resized.astype("float") / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            if debug:
                # Store visualization of normalized face (rescale for visualization)
                debug_output['face_normalized'] = (face_normalized * 255).astype('uint8')
                debug_output['normalization_description'] = (
                    "Pixel values are normalized to the range [0,1] by dividing by 255. "
                    "This standardization helps the neural network process the image values "
                    "consistently regardless of the original pixel value range."
                )
            
            if self.framework == "tensorflow":
                # Make prediction with TensorFlow model
                prediction = self.mask_net.predict(face_batch, verbose=0)[0]
                
                if debug:
                    debug_output['raw_predictions'] = prediction.copy()
                    debug_output['prediction_description'] = (
                        "The preprocessed face is passed through the mask detection model. "
                        "The model outputs confidence scores (logits) for each class: "
                        f"{', '.join([f'{self.class_names[i]}: {p:.4f}' for i, p in enumerate(prediction)])}"
                    )
                
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
                
                if debug:
                    debug_output['thresholds'] = self.thresholds
                    debug_output['selected_class'] = max_index
                    debug_output['threshold_description'] = (
                        f"Applied class-specific thresholds: {self.thresholds}. "
                        f"The model selected '{label}' with confidence {confidence:.4f} "
                        f"(threshold: {self.thresholds[max_index]:.4f}). "
                        "These thresholds help reduce false positives by requiring higher "
                        "confidence for certain classes."
                    )
            
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
                
            if debug:
                return label, color, confidence, debug_output
            else:
                return label, color, confidence
            
        except Exception as e:
            print(f"Error in mask prediction: {e}")
            # Return default values in case of error
            if debug:
                debug_output['error'] = str(e)
                return "Error", (0, 0, 0), 0.0, debug_output
            else:
                return "Error", (0, 0, 0), 0.0

    def process_frame(self, frame, debug=False):
        """
        Process a frame: detect faces, predict mask status, and annotate
        
        If debug=True, returns a list of step-by-step results for visualization
        """
        if frame is None:
            return (None, 0) if not debug else (None, 0, [])
        
        debug_steps = []
        
        # Create a copy to draw on
        output = frame.copy()
        
        # Original Image
        if debug:
            debug_steps.append(
                ("Original Image", frame.copy(), 
                "The raw input image as received by the system. This is the starting point "
                "for all processing. The image may come from a camera feed, video file, or image file.")
            )
        
        # Detect faces
        if debug:
            faces, face_debug = self.detect_faces(frame, debug=True)
            
            # Add all face detection debug steps
            if 'blob_creation' in face_debug:
                debug_steps.append(
                    ("Blob Creation", face_debug['blob_creation'], 
                    face_debug['blob_description'])
                )
            
            if 'resized_frame' in face_debug:
                debug_steps.append(
                    ("Image Resizing", face_debug['resized_frame'], 
                    face_debug['resize_description'])
                )
            
            debug_steps.append(
                ("Face Detection", face_debug.get('detection_visualization'), 
                face_debug.get('detections_description', "") + " " + 
                face_debug.get('face_extraction_description', ""))
            )
        else:
            faces = self.detect_faces(frame)
        
        # Get grayscale version for visualization
        if debug:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            debug_steps.append(
                ("Grayscale Conversion", gray, 
                "Converting the image to grayscale simplifies processing by reducing "
                "the data from 3 channels (BGR) to 1 channel (grayscale). This can help "
                "with certain feature extraction techniques, though our current model "
                "works with color images.")
            )
            
            # Also create an intermediate step showing detected faces
            if len(faces) > 0:
                face_roi_display = output.copy()
                for i, face_data in enumerate(faces):
                    (startX, startY, endX, endY) = face_data["box"]
                    cv2.rectangle(face_roi_display, (startX, startY), (endX, endY), (255, 0, 0), 2)
                    cv2.putText(face_roi_display, f"Face {i+1}", (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                debug_steps.append(
                    ("Face Regions of Interest", face_roi_display, 
                    "Each detected face is extracted as a region of interest (ROI) for "
                    "mask detection processing. The face ROIs are processed independently "
                    "to determine mask status.")
                )
        
        # Process each face
        for i, face_data in enumerate(faces):
            (startX, startY, endX, endY) = face_data["box"]
            face_roi = face_data["roi"]
            
            # Predict mask status
            if debug:
                label, color, confidence, mask_debug = self.predict_mask(face_roi, debug=True)
                
                # Store mask prediction debug info
                debug_steps.append(
                    (f"Face {i+1} RGB Conversion", mask_debug.get('face_rgb'), 
                    mask_debug.get('rgb_description', ""))
                )
                
                debug_steps.append(
                    (f"Face {i+1} Resizing", mask_debug.get('face_resized'), 
                    mask_debug.get('resize_description', ""))
                )
                
                debug_steps.append(
                    (f"Face {i+1} Normalization", mask_debug.get('face_normalized'), 
                    mask_debug.get('normalization_description', ""))
                )
                
                # Raw predictions visualization
                if 'raw_predictions' in mask_debug:
                    # Create a bar chart visualization of predictions
                    pred_vis = np.zeros((200, 300, 3), dtype=np.uint8) + 255  # White background
                    
                    bar_width = 60
                    gap = 30
                    max_height = 150
                    
                    # Draw bars for each class
                    for j, (class_name, pred_value) in enumerate(zip(self.class_names, mask_debug['raw_predictions'])):
                        # Calculate bar position and height
                        x = j * (bar_width + gap) + gap
                        bar_height = int(pred_value * max_height)
                        y = 180 - bar_height  # Bottom aligned
                        
                        # Set color based on class
                        if j == 0:  # With mask
                            bar_color = (0, 255, 0)  # Green
                        elif j == 1:  # Without mask
                            bar_color = (0, 0, 255)  # Red
                        else:  # Incorrect mask
                            bar_color = (0, 255, 255)  # Yellow
                            
                        # Draw the bar
                        cv2.rectangle(pred_vis, (x, y), (x + bar_width, 180), bar_color, -1)
                        
                        # Add class name and value
                        cv2.putText(pred_vis, class_name[:5], (x, 195), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(pred_vis, f"{pred_value:.2f}", (x, y - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    
                    debug_steps.append(
                        (f"Face {i+1} Raw Predictions", pred_vis, 
                        mask_debug.get('prediction_description', "") + " " +
                        mask_debug.get('threshold_description', ""))
                    )
                
            else:
                label, color, confidence = self.predict_mask(face_roi)
            
            # Draw bounding box
            cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
            
            # Create label text with confidence
            text = f"{label}: {confidence:.2f}"
            
            # Determine text position
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            # Add text with background for better visibility
            cv2.rectangle(output, (startX, y-20), (startX + len(text)*11, y), color, -1)
            cv2.putText(output, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Final output with annotations
        if debug:
            debug_steps.append(
                ("Final Result", output, 
                f"Final output with {len(faces)} face(s) identified and classified. "
                f"Green indicates proper mask usage, red indicates no mask, and yellow indicates improper mask usage. "
                "The confidence scores represent the model's certainty about each classification.")
            )
            return output, len(faces), debug_steps
        else:
            return output, len(faces)