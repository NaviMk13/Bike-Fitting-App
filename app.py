import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# --- 1. SEITEN-SETUP ---
st.set_page_config(page_title="DIY KI Bike Fitter", layout="wide", page_icon="🚴")
st.title("🚴 DIY AI Bike Fitting Tool")
st.write("Lade ein seitliches Foto hoch, um deine Haltung analysieren zu lassen.")

# --- 2. YOLO-POSE MODELL LADEN ---
# Lädt ein leichtes, stabiles Pose-Modell, das komplett ohne MediaPipe auskommt
@st.cache_resource
def load_model():
    return YOLO("yolov8n-pose.pt")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Fehler beim Laden des Modells: {e}")
    model_loaded = False

# --- 3. HELFER-FUNKTION: WINKELBERECHNUNG ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# --- 4. BILD-UPLOAD & VERARBEITUNG ---
if model_loaded:
    uploaded_file = st.file_uploader("Wähle ein Foto deines Bike-Fittings aus", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        
        # KI-Erkennung ausführen
        results = model(image_np, verbose=False)
        
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # YOLO Keypoints extrahieren
        # Indizes bei YOLO-Pose: Re-Hüfte = 12, Re-Knie = 14, Re-Knöchel = 16
        # (Sollte die linke Körperseite zu sehen sein, wären es: Li-Hüfte = 11, Li-Knie = 13, Li-Knöchel = 15)
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy[0].cpu().numpy() # Alle erkannten Punkte
            
            # Überprüfen, ob die Punkte für die rechte Seite erkannt wurden (Wert > 0)
            if len(keypoints) > 16 and keypoints[12][0] > 0 and keypoints[14][0] > 0 and keypoints[16][0] > 0:
                hip = keypoints[12]
                knee = keypoints[14]
                ankle = keypoints[16]
                side_label = "Rechte"
            # Fallback auf die linke Seite, falls die rechte nicht im Bild ist
            elif len(keypoints) > 15 and keypoints[11][0] > 0 and keypoints[13][0] > 0 and keypoints[15][0] > 0:
                hip = keypoints[11]
                knee = keypoints[13]
                ankle = keypoints[15]
                side_label = "Linke"
            else:
                hip = None
                
            if hip is not None:
                knee_angle = calculate_angle(hip, knee, ankle)
                
                # Linien zeichnen
                draw.line([tuple(hip), tuple(knee)], fill="lime", width=5)
                draw.line([tuple(knee), tuple(ankle)], fill="lime", width=5)
                
                # Gelenke markieren
                for pt in [hip, knee, ankle]:
                    draw.ellipse([pt[0]-8, pt[1]-8, pt[0]+8, pt[1]+8], fill="red")
                
                # Ergebnis anzeigen
                st.image(image, caption=f"Analysierte Haltung ({side_label} Körperseite)", use_container_width=True)
                
                # --- KI AUSWERTUNG ---
                st.header("🚴 Haltungsanalyse & KI-Tipps")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Gemessener Kniewinkel", value=f"{knee_angle:.1f}°", delta="Optimal: 140°-150°")
                    
                with col2:
                    if knee_angle < 140:
                        st.warning("⚠️ Dein Sattel ist wahrscheinlich zu niedrig!")
                        st.write("**Tipp:** Schiebe deinen Sattel in kleinen Schritten (ca. 5mm) nach oben, um den Druck von der Kniescheibe zu nehmen.")
                    elif knee_angle > 150:
                        st.warning("⚠️ Dein Sattel ist wahrscheinlich zu hoch!")
                        st.write("**Tipp:** Stelle den Sattel etwas tiefer, um ein Überstrecken des Beins und Kippen des Beckens zu verhindern.")
                    else:
                        st.success("🎉 Perfekte Sattelhöhe!")
                        st.write("**Tipp:** Dein Kniewinkel liegt im optimalen ergonomischen Bereich (140°-150°). Die Kraftübertragung ist ideal.")
            else:
                st.error("Es wurden nicht genügend Gelenke (Hüfte, Knie, Knöchel) erkannt. Bitte achte darauf, dass du vollständig und von der Seite auf dem Foto zu sehen bist.")
        else:
            st.error("Es wurde keine Person auf dem Foto erkannt.")
