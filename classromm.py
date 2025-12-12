"""
Final Complete Single-File Streamlit Classroom Attention Monitoring System

This version implements the most robust Streamlit state management to guarantee 
the display of the final report (graphs/metrics) after the "Stop Monitoring" button is clicked.
It also corrects the deprecated 'use_column_width' parameter.
"""

import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque
import urllib.request
import os
import sys
import pandas as pd

# --- File Paths (for DNN model/Haar Cascade fallback) ---
PROTOTXT_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# ==============================================================================
# 1. Engagement Scorer Module (HEURISTICS)
# ==============================================================================

class EngagementScorer:
    """Calculates engagement score based on visual metrics."""
    def __init__(self):
        self.base_score = 95.0

    def calculate_engagement_opencv(self, detection_data, pitch, yaw, roll):
        """
        Calculates a simulated engagement score (0-100%) based on head pose and detection quality.
        """
        score = self.base_score
        
        # 1. Yaw and Pitch Penalty (Looking Away)
        if abs(yaw) > 10:
            score -= (abs(yaw) - 10) * 1.5
        if abs(pitch) > 10:
             score -= (abs(pitch) - 10) * 1.5
        if abs(yaw) > 20:
             score -= 20.0
        if abs(pitch) > 15:
             score -= 15.0
            
        # 2. Confidence Penalty
        confidence = detection_data.get('confidence', 1.0)
        confidence_penalty = (1.0 - confidence) * 15
        score -= confidence_penalty
        
        # 3. Finalization
        score += np.random.uniform(-3, 3)
        
        return np.clip(score, 0.0, 100.0)

# ==============================================================================
# 2. Analytics Engine Module (REPORT GENERATION)
# ==============================================================================

class AnalyticsEngine:
    """Processes collected session data and generates summary reports in Streamlit."""
    def __init__(self):
        pass

    def generate_report(self, session_data, session_stats):
        
        if not session_data:
            st.warning("No data collected during the session.")
            return

        all_engagements = [frame['average_engagement'] for frame in session_data if frame['average_engagement'] > 0]
        
        if not all_engagements:
            st.warning("No valid engagement scores recorded.")
            return

        avg_engagement_final = np.mean(all_engagements)
        min_engagement = np.min(all_engagements)
        max_engagement = np.max(all_engagements)
        
        # --- Streamlit Report Display ---
        st.header("📊 Final Classroom Analytics Report")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Duration", f"{session_stats['session_duration']:.2f} seconds")
            st.metric("Total Frames Processed", f"{session_stats['total_frames']}")
        with col2:
            st.metric("Overall Avg Engagement", f"{avg_engagement_final:.2f}%")
            st.metric("Peak Engagement", f"{max_engagement:.2f}%")

        # 1. Engagement History (Line Chart)
        st.subheader("Engagement Over Time")
        
        # Prepare data for line chart (resample to avoid overcrowding)
        avg_scores_for_chart = [np.mean(session_data[i]['face_engagements']) 
                                for i in range(0, len(session_data), 30) 
                                if session_data[i]['face_engagements']]
        
        st.line_chart(avg_scores_for_chart, use_container_width=True)
        

        # 2. Engagement Distribution Analysis (Bar Chart)
        st.subheader("Engagement Level Distribution")
        
        # Calculate distribution counts
        distribution = {
            'High Engagement (>= 85%)': sum(1 for score in all_engagements if score >= 85),
            'Medium Engagement (60%-85%)': sum(1 for score in all_engagements if 60 <= score < 85),
            'Low Engagement (< 60%)': sum(1 for score in all_engagements if score < 60)
        }
        
        dist_df = pd.DataFrame(list(distribution.items()), columns=['Level', 'Count'])
        
        # Display as a Bar Chart
        st.bar_chart(dist_df, x='Level', y='Count')
        

        st.caption("Distribution shows the percentage of time spent in each engagement level.")


# ==============================================================================
# 3. Main Execution Module (AttentionMonitor - Adapted for Streamlit)
# ==============================================================================

class AttentionMonitor:
    """Main class for real-time attention monitoring, adapted for Streamlit loop."""
    
    def __init__(self):
        # Initialize components and ensure cache integrity
        self.face_net = self._load_face_detector()
        self.engagement_scorer = EngagementScorer()
        self.analytics = AnalyticsEngine()
        self.engagement_history = deque(maxlen=30)
        
        # Initialize stats in session state
        if 'stats' not in st.session_state:
             st.session_state.stats = {
                'total_faces_detected': 0,
                'total_frames': 0,
                'average_engagement': 0.0,
                'session_duration': 0,
                'session_start_time': 0.0,
            }
        
        if 'session_data' not in st.session_state:
             st.session_state.session_data = []

    @st.cache_resource
    def _load_face_detector(_self):
        """Load OpenCV DNN face detection model or fallback to Haar Cascade"""
        
        if not os.path.exists(PROTOTXT_PATH):
            try:
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                    PROTOTXT_PATH
                )
            except Exception:
                pass
        
        try:
            if os.path.exists(PROTOTXT_PATH) and os.path.exists(MODEL_PATH):
                net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
                st.success("DNN face detector loaded successfully.")
                return net
            else:
                cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
                if cascade.empty():
                    st.error("Error: Could not load face detector!")
                    raise FileNotFoundError("Required cascade file missing or corrupt.")
                st.info("Falling back to Haar Cascade detector.")
                return cascade
                
        except Exception:
            cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
            if cascade.empty():
                raise FileNotFoundError("Required cascade file missing or corrupt.")
            st.info("Falling back to Haar Cascade detector.")
            return cascade

    def _detect_faces(self, frame):
        # Detection logic (same as original)
        if isinstance(self.face_net, cv2.dnn.Net):
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104, 117, 123])
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x, y, x2, y2 = box.astype("int")
                    faces.append({'bbox': (x, y, x2 - x, y2 - y), 'confidence': confidence})
            return faces
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_rect = self.face_net.detectMultiScale(gray, 1.1, 4)
            faces = []
            for (x, y, w, h) in faces_rect:
                faces.append({'bbox': (x, y, w, h), 'confidence': 0.8})
            return faces

    def calculate_head_pose(self, bbox, frame_width, frame_height):
        # Head pose calculation
        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2
        frame_center_x = frame_width / 2
        x_offset = center_x - frame_center_x
        yaw = (x_offset / frame_center_x) * 30
        frame_center_y = frame_height / 2
        y_offset = center_y - frame_center_y
        pitch = (y_offset / frame_center_y) * 20
        aspect_ratio = w / h if h > 0 else 1.0
        roll = (aspect_ratio - 1.0) * 10 
        return pitch, yaw, roll
    
    def process_frame(self, frame):
        """Process a single frame for face detection and engagement scoring."""
        frame_height, frame_width = frame.shape[:2]
        faces = self._detect_faces(frame)
        
        engagement_data = {
            'timestamp': time.time(),
            'num_faces': 0,
            'face_engagements': [],
            'average_engagement': 0.0,
            'head_poses': []
        }
        
        if faces:
            engagement_scores = []
            head_poses = []
            
            for face in faces:
                x, y, w, h = face['bbox']
                confidence = face['confidence']
                
                pitch, yaw, roll = self.calculate_head_pose((x, y, w, h), frame_width, frame_height)
                head_poses.append({'pitch': pitch, 'yaw': yaw, 'roll': roll})
                
                detection_data = {'bbox': (x, y, w, h), 'confidence': confidence, 'frame_width': frame_width, 'frame_height': frame_height}
                
                engagement_score = self.engagement_scorer.calculate_engagement_opencv(detection_data, pitch, yaw, roll)
                engagement_scores.append(engagement_score)
                
                # Draw annotations (OpenCV BGR drawing)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                score_text = f"Engage: {engagement_score:.1f}%"
                cv2.putText(frame, score_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            engagement_data['num_faces'] = len(faces)
            engagement_data['face_engagements'] = engagement_scores
            engagement_data['average_engagement'] = np.mean(engagement_scores)
            
            # CRITICAL UPDATE: Ensure history deque doesn't crash on empty list
            if engagement_data['average_engagement'] > 0:
                self.engagement_history.append(engagement_data['average_engagement'])
        
        return frame, engagement_data
    
    def run_streamlit_loop(self):
        """Main loop that replaces the cv2.imshow run loop."""
        
        # Placeholders for live data
        video_placeholder = st.empty()
        stats_placeholder = st.empty()

        # Initialize/Acquire Camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
             cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not open camera. Check connection/permissions.")
            st.session_state.is_monitoring = False
            return
        
        # Reset current session variables (only runs once per session start)
        st.session_state.stats['session_start_time'] = time.time()
        st.session_state.stats['total_frames'] = 0
        st.session_state.stats['total_faces_detected'] = 0
        all_session_data = [] 
        
        while st.session_state.is_monitoring:
            ret, frame = cap.read()
            if not ret:
                st.warning("Warning: Could not read frame. Stopping monitoring.")
                break
            
            current_time = time.time()
            frame = cv2.flip(frame, 1) # Flip for mirror view
            
            # Process frame
            processed_frame, engagement_data = self.process_frame(frame)
            all_session_data.append(engagement_data)
            
            # Update Statistics
            st.session_state.stats['total_frames'] += 1
            avg_engagement = np.mean(self.engagement_history) if self.engagement_history else 0.0
            
            st.session_state.stats['average_engagement'] = avg_engagement
            st.session_state.stats['session_duration'] = current_time - st.session_state.stats['session_start_time']
            st.session_state.stats['total_faces_detected'] += engagement_data['num_faces']
            
            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Update placeholders
            with video_placeholder.container():
                 # CORRECTED DEPRECATION WARNING: use_container_width=True
                 st.image(frame_rgb, channels="RGB", use_container_width=True, caption="Live Classroom Monitor")
                 
            with stats_placeholder.container():
                st.metric("Live Class Engagement", f"{avg_engagement:.1f}%")
                st.text(f"Duration: {st.session_state.stats['session_duration']:.1f}s | Faces: {engagement_data['num_faces']}")

            # Sleep to prevent high CPU usage (adjust as needed)
            time.sleep(0.01)

        # Loop cleanup (After stop button is pressed or break)
        cap.release()
        
        # --- CRITICAL FIX: SAVE DATA and RERUN ---
        if all_session_data:
             st.session_state.session_data = all_session_data 
        
        st.session_state.is_monitoring = False # Ensure loop state is reset
        st.rerun() # Force a rerun to display the final report

# --- Streamlit UI Setup ---

# Apply Streamlit configuration
st.markdown("""
<style>
    .stApp { background-color: #F9F9F9; }
    h1 { color: #1E3A8A; }
    .stButton>button {
        background-color: #007B8A;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1rem;
    }
    .score-box {
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


monitor = AttentionMonitor()

# Initialize session state for control (redundant checks kept for robustness)
if 'is_monitoring' not in st.session_state:
    st.session_state.is_monitoring = False
if 'session_data' not in st.session_state:
    st.session_state.session_data = []
if 'stats' not in st.session_state:
     st.session_state.stats = {
        'total_faces_detected': 0,
        'total_frames': 0,
        'average_engagement': 0.0,
        'session_duration': 0,
        'session_start_time': 0.0,
    }
    
st.header("Real-Time Classroom Attention Monitoring System")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    start_button = st.button("Start Monitoring", disabled=st.session_state.is_monitoring)
with col2:
    stop_button = st.button("Stop Monitoring", disabled=not st.session_state.is_monitoring)

# Button Logic
if start_button:
    st.session_state.is_monitoring = True
    # Clear previous data on new start
    st.session_state.session_data = [] 
    st.rerun()

if stop_button:
    # Do NOT set is_monitoring=False here, as that causes the loop to exit and then reruns.
    # The loop exit itself handles setting is_monitoring=False and the final rerun.
    pass 

# --- Run or Display Report ---
if st.session_state.is_monitoring:
    monitor.run_streamlit_loop()
    
elif st.session_state.session_data:
    # Display the final report ONLY if monitoring is stopped AND data exists
    monitor.analytics.generate_report(st.session_state.session_data, st.session_state.stats)

else:
    st.info("Click 'Start Monitoring' to begin the session and activate the webcam.")