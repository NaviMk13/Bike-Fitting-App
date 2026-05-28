import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import av
import tempfile
import os

# --- 1. DESIGN & FAHRRAD-VIBE (CUSTOM CSS) ---
st.set_page_config(page_title="VELO-MATCH KI Bike Fitter", layout="wide", page_icon="🚴")

st.markdown("""
    <style>
    /* Hintergrundbild mit Radsport-Atmosphäre */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.95)), 
                    url('https://images.unsplash.com/photo-1485965120184-e220f721d03e?q=80&w=1920') no-repeat center center fixed;
        background-size: cover;
        color: #f8fafc;
    }
    
    /* Titel-Styling */
    h1 {
        font-family: 'Impact', 'Arial Black', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #facc15 !important; /* Neon-Gelb */
        text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
    }
    
    /* Karten für Ergebnisse */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 2px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(8px);
        margin-bottom: 15px;
    }
    
    /* Fahrrad-Animation beim Laden */
    @keyframes ride {
        0% { transform: translateX(-20px) rotate(0deg); }
        50% { transform: translateX(20px) rotate(3deg); }
        100% { transform: translateX(-20px) rotate(0deg); }
    }
    .bike-loader {
        font-size: 50px;
        animation: ride 2s infinite ease-in-out;
        text-align: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True) # HIER WAR DER FEHLER (jetzt korrekt unsafe_allow_html)

st.title("🚴 VELO-MATCH: AI DYNAMIC BIKE FITTING")
st.write("Optimiere deine Aero-Position und Ergonomie. Lade deine seitliche Videoaufnahme hoch.")

# --- 2. KI-MODELL INITIALISIERUNG ---
@st.cache_resource
def load_pose_model():
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

# --- 3. MATHEMATISCH KORREKTE WINKELBERECHNUNG ---
def calculate_angle(a, b, c, interior=True):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    if not interior:
        return angle # Außenwinkel (z.B. Kniewinkel ~140°-145°)
    else:
        # Berechnet den echten Beugewinkel aus der Streckung heraus (z.B. Armbeugung ~20°)
        return 180.0 - angle if angle > 90 else angle

# --- 4. DYNAMISCHE ANALYSE ---
if model_loaded:
    uploaded_file = st.file_uploader("📂 Ziehe dein Video hierher (.mp4, .mov)", type=["mp4", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Coole Lade-Animation anzeigen
        status_text = st.empty()
        loader_anim = st.empty()
        status_text.info("🚴 Analysiere Trittzyklus und Aerodynamik... Bitte warten.")
        loader_anim.markdown("<div class='bike-loader'>🚴💨</div>", unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        
        container = av.open(tfile.name)
        stream = container.streams.video[0]
        
        frames_data = []
        processed_images = []
        
        total_frames = stream.frames if stream.frames > 0 else 100
        current_frame_idx = 0
        
        for frame in container.decode(video=0):
            # Analysiere jeden 3. Frame für flüssige Verarbeitung ohne Server-Timeout
            if current_frame_idx % 3 == 0:
                img = frame.to_image().convert("RGB")
                input_tensor = data_transforms(img).unsqueeze(0)
                
                with torch.no_grad():
                    prediction = model(input_tensor)[0]
                
                if len(prediction['keypoints']) > 0 and prediction['scores'][0] > 0.8:
                    kp = prediction['keypoints'][0].cpu().numpy()
                    scores = prediction['keypoints_scores'][0].cpu().numpy()
                    
                    # Körperseite bestimmen (Rechts vs. Links)
                    if scores[14] > scores[13]:
                        side = "Rechte Seite"
                        h, k, a = kp[12][:2], kp[14][:2], kp[16][:2] # Hüfte, Knie, Knöchel
                        s, e, w = kp[6][:2], kp[8][:2], kp[10][:2]   # Schulter, Ellbogen, Handgelenk
                    else:
                        side = "Linke Seite"
                        h, k, a = kp[11][:2], kp[13][:2], kp[15][:2]
                        s, e, w = kp[5][:2], kp[7][:2], kp[9][:2]
                    
                    # Winkel berechnen
                    knee_angle = calculate_angle(h, k, a, interior=False)
                    hip_angle = calculate_angle(s, h, k, interior=False)
                    arm_angle = calculate_angle(s, e, w, interior=True)   
                    shoulder_angle = calculate_angle(h, s, e, interior=False)
                    
                    # Zeichnen der Overlays auf dem aktuellen Frame
                    draw_img = img.copy()
                    draw = ImageDraw.Draw(draw_img)
                    
                    # Linien für Beine (Grün) und Cockpit (Neon-Gelb)
                    draw.line([tuple(h), tuple(k)], fill="#22c55e", width=5)
                    draw.line([tuple(k), tuple(a)], fill="#22c55e", width=5)
                    draw.line([tuple(s), tuple(h)], fill="#06b6d4", width=4) # Rumpf
                    draw.line([tuple(s), tuple(e)], fill="#eab308", width=4) # Oberarm
                    draw.line([tuple(e), tuple(w)], fill="#eab308", width=4) # Unterarm
                    
                    # Gelenkpunkte markieren
                    for pt in [h, k, a, s, e, w]:
                        draw.ellipse([pt[0]-6, pt[1]-6, pt[0]+6, pt[1]+6], fill="#ef4444")
                    
                    processed_images.append(draw_img)
                    
                    frames_data.append({
                        'knee': knee_angle,
                        'hip': hip_angle,
                        'arm': arm_angle,
                        'shoulder': shoulder_angle,
                        'side': side
                    })
            
            current_frame_idx += 1
            progress_bar.progress(min(current_frame_idx / total_frames, 1.0))
            if current_frame_idx > 150: # Schutz vor Überlastung
                break
                
        container.close()
        os.unlink(tfile.name)
        
        # Loader entfernen
        status_text.empty()
        loader_anim.empty()
        
        if frames_data and processed_images:
            st.
