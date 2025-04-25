import streamlit as st
import cv2
import time
import numpy as np
from PIL import Image
import io

from script.streamlit_detector import StreamlitDetector

# Set page config
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# Initialize session state variables if they don't exist
if 'detector' not in st.session_state:
    st.session_state.detector = StreamlitDetector(width=480, height=360)
    st.session_state.detection_active = False
    st.session_state.debug_mode = False

# Set up the app
st.title("Real-time Face Mask Detection")

# App description
st.markdown("""
This application detects face masks in real-time video from your webcam.
- **Green**: Properly wearing a mask
- **Red**: Not wearing a mask
- **Yellow**: Improperly wearing a mask
""")

# Debug/Go Live toggle button
if st.button("Debug Mode" if not st.session_state.debug_mode else "Go Live"):
    # Toggle debug mode
    st.session_state.debug_mode = not st.session_state.debug_mode
    # Reset detection if active
    if st.session_state.detection_active:
        st.session_state.detector.stop()
        st.session_state.detection_active = False
    st.rerun()

# Live detection mode
if not st.session_state.debug_mode:
    # Check if detector initialized properly
    if st.session_state.detector.detector is None:
        st.error("Failed to initialize face mask detection model. Check console for details.")
    else:
        st.info("Face mask detection model loaded successfully. Click 'Start Detection' to begin.")
    
    # Create columns for buttons
    col1, col2 = st.columns(2)

    # Add control buttons with disabled state based on detection status
    if col1.button("Start Detection", disabled=st.session_state.detection_active):
        st.session_state.detector.start()
        st.session_state.detection_active = True
        st.rerun()

    if col2.button("Stop Detection", disabled=not st.session_state.detection_active):
        st.session_state.detector.stop()
        st.session_state.detection_active = False
        st.rerun()

    # Placeholder for video frame with controlled width
    frame_placeholder = st.empty()

    # Status indicator
    status = st.empty()

    # Check if detection is running
    if st.session_state.detection_active:
        status.success("Camera is active")
        st.markdown("### Live Detection View")
        
        # Run in a Streamlit loop
        while st.session_state.detector.running:
            frame = st.session_state.detector.get_frame()
            if frame is not None:
                # Convert BGR to RGB for display in Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Update the frame placeholder with controlled width
                frame_placeholder.image(
                    frame_rgb, 
                    channels="RGB", 
                    use_container_width=True  # This helps control the width
                )
            
            # Short delay to prevent high CPU usage
            time.sleep(0.1)

# Debug mode
else:
    st.subheader("Debug Mode: Step-by-Step Face Mask Detection")
    st.info("Upload an image to see step-by-step processing of how the face mask detection works.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        image_bytes = uploaded_file.getvalue()
        pil_image = Image.open(io.BytesIO(image_bytes))
        input_image = np.array(pil_image)
        
        # Convert RGB to BGR (OpenCV format)
        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        
        # Process image step by step and display the results
        if st.session_state.detector.detector is not None:
            debug_results = st.session_state.detector.process_image_debug(input_image)
            
            # Display original image
            st.subheader("1. Original Image")
            st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Display each step from debug results
            for i, (title, image, description) in enumerate(debug_results, start=2):
                st.subheader(f"{i}. {title}")
                if image is not None:
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        # Convert BGR to RGB for display if it's a color image
                        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        # If grayscale, no conversion needed
                        display_image = image
                    st.image(display_image,width=200)
                st.write(description)
        
        else:
            st.error("Face mask detector is not initialized properly.")
    
    # Display warning about performance
    st.warning("Note: Due to Streamlit's execution model, there might be some latency. For better performance, consider running this application locally.")