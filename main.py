import streamlit as st
import cv2
import numpy as np
import time
from live import StreamlitDetector

# Set page config
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"  # Changed from "wide" to "centered" for smaller frame
)

# Set up the app
st.title("Real-time Face Mask Detection")

# App description
st.markdown("""
This application detects face masks in real-time video from your webcam.
- **Green**: Properly wearing a mask
- **Red**: Not wearing a mask
- **Yellow**: Improperly wearing a mask
""")

# Initialize detector and button states in session state if they don't exist
if 'detector' not in st.session_state:
    st.session_state.detector = StreamlitDetector(width=480, height=360)  # Reduced size
    st.session_state.detection_active = False
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
        
    # Display warning about performance
    st.warning("Note: Due to Streamlit's execution model, there might be some latency. For better performance, consider running this application locally.")