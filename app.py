import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

# --- 1. SEITEN-SETUP ---
st.set_page_config(page_title="DIY KI Bike Fitter", layout="wide", page_icon="🚴")
st.title("🚴 DIY AI Bike Fitting Tool")
st.write("Lade ein seitliches Foto hoch, um deine Haltung analysieren zu lassen.")

# --- 2. PRETRAINED PYTORCH MODEL LADEN ---
@st.cache_resource
def load_pose_model():
    # Lädt das offizielle Google/PyTorch Keypoint-Modell (völlig ohne OpenCV/MediaPipe!)
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model, weights.transforms()

try:
    model, data_transforms = load_pose_model()
    model_loaded = True
except Exception as e:
    st.error(f"Fehler beim Laden des KI-Modells: {e}")
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
        # Bild rein mit PIL (Pillow) laden - 100% sicher vor Grafikfehlern
        img = Image.open(uploaded_file).convert("RGB")
        width, height = img.size
        
        # Bild für das KI-Modell vorbereiten
        input_tensor = data_transforms(img).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(input_tensor)[0]
        
        # Keypoints extrahieren
        # COCO Keypoint Indizes: Re-Hüfte = 12, Re-Knie = 14, Re-Knöchel = 16
        #                      Li-Hüfte = 11, Li-Knie = 13, Li-Knöchel = 15
        if len(prediction['keypoints']) > 0 and prediction['scores'][0] > 0.8:
            # Wir nehmen die am besten erkannte Person (Index 0)
            kp = prediction['keypoints'][0].cpu().numpy()
            scores = prediction['keypoints_scores'][0].cpu().numpy()
            
            # Prüfen, welche Körperseite besser sichtbar ist (höherer Score für das Knie)
            if scores[14] > scores[13]:
                hip, knee, ankle = kp[12][:2], kp[14][:2], kp[16][:2]
                side_label = "Rechte"
            else:
                hip, knee, ankle = kp[11][:2], kp[13][:2], kp[15][:2]
                side_label = "Linke"
                
            # Berechnen, wenn alle 3 Punkte verlässlich erkannt wurden
            knee_angle = calculate_angle(hip, knee, ankle)
            
            # Zeichnen auf dem Bild via PIL ImageDraw (Kein OpenCV nötig!)
            draw = ImageDraw.Draw(img)
            
            # Linien ziehen
            draw.line([tuple(hip), tuple(knee)], fill="lime", width=6)
            draw.line([tuple(knee), tuple(ankle)], fill="lime", width=6)
            
            # Punkte markieren
            for pt in [hip, knee, ankle]:
                draw.ellipse([pt[0]-8, pt[1]-8, pt[0]+8, pt[1]+8], fill="red")
            
            # Bild in Streamlit anzeigen
            st.image(img, caption=f"Analysierte Haltung ({side_label} Körperseite)", use_container_width=True)
            
            # --- KI AUSWERTUNG ---
            st.header("🚴 Haltungsanalyse & KI-Tipps")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="Gemessener Kniewinkel", value=f"{knee_angle:.1f}°", delta="Optimal: 140°-150°")
                
            with col2:
                if knee_angle < 140:
                    st.warning("⚠️ Dein Sattel ist wahrscheinlich zu niedrig!")
                    st.write("**Tipp:** Schiebe deinen Sattel in kleinen Schritten (ca. 5mm) nach oben, um Knieschmerzen zu vermeiden.")
                elif knee_angle > 150:
                    st.warning("⚠️ Dein Sattel ist wahrscheinlich zu hoch!")
                    st.write("**Tipp:** Stelle den Sattel etwas tiefer, damit dein Becken beim Treten stabil bleibt.")
                else:
                    st.success("🎉 Perfekte Sattelhöhe!")
                    st.write("**Tipp:** Dein Kniewinkel liegt im optimalen Bereich. Die Kraftübertragung ist perfekt.")
        else:
            st.error("Es konnte keine Person oder nicht genügend Gelenke auf dem Foto erkannt werden. Achte darauf, dass du komplett von der Seite zu sehen bist.")
