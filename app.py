import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageDraw
import os
import urllib.request

# --- 1. SEITEN-SETUP ---
st.set_page_config(page_title="DIY KI Bike Fitter", layout="wide", page_icon="🚴")
st.title("🚴 DIY AI Bike Fitting Tool")
st.write("Lade ein seitliches Foto deines Fahrrads hoch, um deine Haltung analysieren zu lassen.")

# --- 2. MODELL AUTOMATISCH VON GOOGLE LADEN ---
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_PATH = "pose_landmarker.task"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Lade KI-Modell herunter... Bitte kurz warten."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# --- 3. HELFER-FUNKTION: WINKELBERCHNUNG ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# --- 4. BILD-UPLOAD ---
uploaded_file = st.file_uploader("Wähle ein Foto deines Bike-Fittings aus", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild mit PIL öffnen
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    # MediaPipe Task vorbereiten
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        output_segmentation_masks=False
    )
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        # Konvertiere in das MediaPipe eigene Bildformat
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        
        # Erkennung ausführen
        detection_result = landmarker.detect(mp_image)
        
        # Zeichnen vorbereiten via PIL
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            # Wir extrahieren die Landmarks für die rechte Seite
            # Indizes: Hüfte = 24, Knie = 26, Knöchel = 28
            landmarks = detection_result.pose_landmarks[0]
            
            hip_lm = landmarks[24]
            knee_lm = landmarks[26]
            ankle_lm = landmarks[28]
            
            # Pixelkoordinaten berechnen
            hip = [hip_lm.x * width, hip_lm.y * height]
            knee = [knee_lm.x * width, knee_lm.y * height]
            ankle = [ankle_lm.x * width, ankle_lm.y * height]
            
            # Kniewinkel berechnen
            knee_angle = calculate_angle(hip, knee, ankle)
            
            # Linien auf das Bild zeichnen
            draw.line([tuple(hip), tuple(knee)], fill="lime", width=5)
            draw.line([tuple(knee), tuple(ankle)], fill="lime", width=5)
            
            # Punkte markieren
            for pt in [hip, knee, ankle]:
                draw.ellipse([pt[0]-8, pt[1]-8, pt[0]+8, pt[1]+8], fill="red")
            
            # Bild anzeigen
            st.image(image, caption="Analysierte Haltung", use_container_width=True)
            
            # --- KI AUSWERTUNG ---
            st.header("🚴 Haltungsanalyse & KI-Tipps")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="Gemessener maximaler Kniewinkel", value=f"{knee_angle:.1f}°", delta="Optimal: 140°-150°")
                
            with col2:
                if knee_angle < 140:
                    st.warning("⚠️ Dein Sattel ist wahrscheinlich zu niedrig!")
                    st.write("**Tipp:** Schiebe deinen Sattel in kleinen Schritten (ca. 5mm) nach oben.")
                elif knee_angle > 150:
                    st.warning("⚠️ Dein Sattel ist wahrscheinlich zu hoch!")
                    st.write("**Tipp:** Stelle den Sattel etwas tiefer.")
                else:
                    st.success("🎉 Perfekte Sattelhöhe!")
                    st.write("**Tipp:** Dein Kniewinkel liegt genau im ergonomischen Bereich.")
        else:
            st.error("Es wurden keine Gelenke im Bild erkannt. Bitte achte darauf, dass du ganz auf dem Foto zu sehen bist.")
