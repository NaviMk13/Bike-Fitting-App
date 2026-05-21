# --- 0. AUTOMATISCHER SERVER-HOTFIX (MUSS GANZ OBEN STEHEN) ---
import sys
import subprocess

try:
    import cv2
except ImportError:
    # Falls das normale OpenCV wegen MediaPipe-Konflikten oder fehlender 
    # libGL.so.1 abstürzt, bereinigen wir die Umgebung vollautomatisch.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", 
                    "opencv-python", "opencv-contrib-python", 
                    "opencv-python-headless", "opencv-contrib-python-headless"])
    subprocess.run([sys.executable, "-m", "pip", "install", "opencv-contrib-python-headless"])
    import cv2

# --- AB HIER FOLGT DER NORMALE CODE ---
import streamlit as st
import mediapipe as mp
import numpy as np
import tempfile
import os

# --- 1. SEITEN-SETUP ---
st.set_page_config(page_title="DIY KI Bike Fitter", layout="wide", page_icon="🚴")
st.title("🚴 DIY AI Bike Fitting Tool")
st.write("Lade ein seitliches Video von dir auf dem Fahrrad hoch, um deine Haltung analysieren zu lassen.")

# --- 2. MEDIAPIPE INITIALISIERUNG ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 3. HELFER-FUNKTION: WINKELBERCHNUNG ---
def calculate_angle(a, b, c):
    """Berechnet den Winkel am Punkt B aus drei Punkten A, B und C."""
    a = np.array(a)  # Start
    b = np.array(b)  # Scheitelpunkt (Knie)
    c = np.array(c)  # Ende
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

# --- 4. VIDEO UPLOAD & VERARBEITUNG ---
uploaded_file = st.file_uploader("Wähle ein Video aus (mp4, mov)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) else 30
    
    st.info("Video wird analysiert... Bitte kurz warten.")
    
    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_knee_angles = []
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(out_tfile.name, fourcc, fps, (width, height))
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.get()
            if not ret:
                break
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            results = pose.process(image)
            
            image.flags.writeable = True
            annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Rechte Körperseite messen
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                knee_angle = calculate_angle(hip, knee, ankle)
                all_knee_angles.append(knee_angle)
                
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                cv2.putText(annotated_image, f"Knie: {int(knee_angle)} Grad", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            out.write(annotated_image)
            
            current_frame += 1
            if frame_count > 0:
                progress_bar.progress(min(current_frame / frame_count, 1.0))
                
        cap.release()
        out.release()
    
    st.success("Analyse abgeschlossen!")
    
    # --- 5. KI-AUSWERTUNG & TIPPS ---
    if all_knee_angles:
        max_knee_angle = max(all_knee_angles)
        
        st.header("🚴 Haltungsanalyse & KI-Tipps")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Gemessener maximaler Kniewinkel", value=f"{max_knee_angle:.1f}°", delta="Optimal: 140°-150°")
            
        with col2:
            if max_knee_angle < 140:
                st.warning("⚠️ Dein Sattel ist wahrscheinlich zu niedrig!")
                st.write("**Tipp:** Wenn der Kniewinkel zu spitz ist, werden deine Knie an der Vorderseite stark belastet. Schiebe deinen Sattel in kleinen Schritten (ca. 5mm) nach oben.")
            elif max_knee_angle > 150:
                st.warning("⚠️ Dein Sattel ist wahrscheinlich zu hoch!")
                st.write("**Tipp:** Wenn das Bein zu stark gestreckt wird, muss dein Becken kippen. Das sorgt für Instabilität. Stelle den Sattel etwas tiefer.")
            else:
                st.success("🎉 Perfekte Sattelhöhe!")
                st.write("**Tipp:** Dein Kniewinkel liegt genau im ergonomischen Bereich. Deine Kraftübertragung ist optimal.")
                
    os.unlink(tfile.name)
    os.unlink(out_tfile.name)
