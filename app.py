import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import av
import tempfile
import os

# --- 1. SEITEN-SETUP ---
st.set_page_config(page_title="DIY KI Bike Fitter Pro", layout="wide", page_icon="🚴")
st.title("🚴 DIY AI Pro Bike Fitting Tool")
st.write("Lade ein seitliches Video hoch. Die KI analysiert deine Gelenkwinkel über die gesamte Kurbellumdrehung.")

# --- 2. KI-MODELL LADEN ---
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

# --- 4. VIDEO-UPLOAD & BIOMECHANIK-ANALYSE ---
if model_loaded:
    uploaded_file = st.file_uploader("Wähle dein Bike-Fitting Video aus (mp4, mov)", type=["mp4", "mov"])

    if uploaded_file is not None:
        # Temporäre Datei für das Video erstellen
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        
        st.info("Video geladen. Starte dynamische Trittzyklus-Analyse... Bitte warten.")
        progress_bar = st.progress(0)
        
        # Video mit PyAV öffnen (Reines Python, kein OpenCV-Absturz!)
        container = av.open(tfile.name)
        stream = container.streams.video[0]
        
        frames_data = []
        all_knee_angles = []
        best_frames = {}
        
        total_frames = stream.frames if stream.frames > 0 else 150
        current_frame_idx = 0
        
        # Wir tasten das Video ab (z.B. jeden 2. Frame für bessere Performance)
        for frame in container.decode(video=0):
            if current_frame_idx % 2 == 0:
                img = frame.to_image().convert("RGB")
                input_tensor = data_transforms(img).unsqueeze(0)
                
                with torch.no_grad():
                    prediction = model(input_tensor)[0]
                
                # COCO Keypoints: 5=Li-Schulter, 6=Re-Schulter, 7=Li-Ellbogen, 8=Re-Ellbogen, 
                # 9=Li-Hand, 10=Re-Hand, 11=Li-Hüfte, 12=Re-Hüfte, 13=Li-Knie, 14=Re-Knie, 15=Li-Knöchel, 16=Re-Knöchel
                if len(prediction['keypoints']) > 0 and prediction['scores'][0] > 0.8:
                    kp = prediction['keypoints'][0].cpu().numpy()
                    scores = prediction['keypoints_scores'][0].cpu().numpy()
                    
                    # Bestimmen, welche Körperseite der Kamera zugewandt ist
                    if scores[14] > scores[13]: # Rechte Seite dominanter
                        side = "Rechte"
                        h, k, a = kp[12][:2], kp[14][:2], kp[16][:2] # Hüfte, Knie, Knöchel
                        s, e, w = kp[6][:2], kp[8][:2], kp[10][:2]   # Schulter, Ellbogen, Handgelenk
                    else:
                        side = "Linke"
                        h, k, a = kp[11][:2], kp[13][:2], kp[15][:2]
                        s, e, w = kp[5][:2], kp[7][:2], kp[9][:2]
                    
                    # Winkel berechnen
                    knee_angle = calculate_angle(h, k, a)
                    hip_angle = calculate_angle(s, h, k)
                    arm_angle = calculate_angle(s, e, w)
                    shoulder_angle = calculate_angle(h, s, e)
                    
                    all_knee_angles.append(knee_angle)
                    
                    frames_data.append({
                        'frame_idx': current_frame_idx,
                        'img': img,
                        'knee': knee_angle,
                        'hip': hip_angle,
                        'arm': arm_angle,
                        'shoulder': shoulder_angle,
                        'points': [h, k, a, s, e, w],
                        'side': side
                    })
            
            current_frame_idx += 1
            if current_frame_idx <= total_frames:
                progress_bar.progress(min(current_frame_idx / total_frames, 1.0))
            if current_frame_idx > 300: # Limitieren, um Timeout auf dem Server zu verhindern
                break
                
        container.close()
        os.unlink(tfile.name)
        
        if frames_data:
            st.success("Dynamische Gelenkanalyse abgeschlossen!")
            
            # Finde die Extrempunkte im Trittzyklus
            # 1. Maximale Beinstreckung (Pedal ganz unten, ca. 5-6 Uhr)
            max_ext_idx = np.argmax([f['knee'] for f in frames_data])
            frame_max_ext = frames_data[max_ext_idx]
            
            # 2. Maximale Beugung / Kraftphase (Pedal oben bzw. bei 3 Uhr - Knie angewinkelter)
            min_ext_idx = np.argmin([f['knee'] for f in frames_data])
            frame_min_ext = frames_data[min_ext_idx]
            
            # --- VISUALISIERUNG DER STRECKUNG (5-6 UHR) ---
            st.header(f"📊 Analyse-Ergebnisse ({frame_max_ext['side']} Körperseite)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("1. Maximale Beinstreckung (Pedal unten)")
                img_ext = frame_max_ext['img'].copy()
                draw = ImageDraw.Draw(img_ext)
                h, k, a, s, e, w = frame_max_ext['points']
                
                # Linien zeichnen
                draw.line([tuple(h), tuple(k)], fill="lime", width=6)
                draw.line([tuple(k), tuple(a)], fill="lime", width=6)
                draw.line([tuple(s), tuple(h)], fill="cyan", width=4)
                draw.line([tuple(s), tuple(e)], fill="orange", width=4)
                draw.line([tuple(e), tuple(w)], fill="orange", width=4)
                
                st.image(img_ext, caption="Erkanntes Maximum der Beinstreckung", use_container_width=True)
                
                st.metric(label="Maximaler Kniewinkel (Sattelhöhe)", value=f"{frame_max_ext['knee']:.1f}°", delta="Optimal: 140°-145°")
                st.metric(label="Hüftwinkel in Streckung", value=f"{frame_max_ext['hip']:.1f}°", delta="Optimal: ~45°")
                
            with col2:
                st.subheader("2. Ergonomie der Arm- & Oberkörperhaltung")
                img_arm = frame_max_ext['img'].copy()
                draw_arm = ImageDraw.Draw(img_arm)
                
                # Fokus auf Cockpit zeichnen
                draw_arm.line([tuple(s), tuple(e)], fill="orange", width=6)
                draw_arm.line([tuple(e), tuple(w)], fill="orange", width=6)
                draw_arm.line([tuple(h), tuple(s)], fill="cyan", width=6)
                
                st.image(img_arm, caption="Analyse der Cockpit-Ergonomie", use_container_width=True)
                
                st.metric(label="Ellbogenwinkel (Armbeugung)", value=f"{frame_max_ext['arm']:.1f}°", delta="Optimal: 15°-25° (leicht gebeugt)")
                st.metric(label="Schulterwinkel", value=f"{frame_max_ext['shoulder']:.1f}°", delta="Optimal: 80°-90°")

            # --- EXPERTEN KI-AUSWERTUNG ---
            st.header("🚴 Automatisierte Bike-Fitting Empfehlungen")
            
            # Sattelhöhen-Logik
            k_angle = frame_max_ext['knee']
            if k_angle < 138:
                st.warning("⚠️ **Sattelhöhe:** Dein Sattel ist deutlich **zu niedrig**.")
                st.write("*Empfehlung:* Schiebe den Sattel um ca. 5-10 mm nach oben. Ein zu niedriger Sattel kostet Performance und belastet die Kniescheibe.")
            elif k_angle > 147:
                st.warning("⚠️ **Sattelhöhe:** Dein Sattel ist etwas **zu hoch**.")
                st.write("*Empfehlung:* Stelle den Sattel etwas tiefer. Ein zu hoher Sattel führt zum Kippen des Beckens und Überstrecken der Sehnen in der Kniekehle.")
            else:
                st.success("🎉 **Sattelhöhe:** Dein Kniewinkel bei maximaler Streckung ist im **optimalen Bereich (140°-145°)**!")
            
            # Cockpit/Vorbau-Logik
            s_angle = frame_max_ext['shoulder']
            if s_angle > 95:
                st.error("⚠️ **Lenker-Reach / Vorbau:** Du sitzt **zu gestreckt** (Schulterwinkel zu groß).")
                st.write("*Empfehlung:* Dein Vorbau ist vermutlich zu lang oder dein Lenker hat zu viel Reach. Ein kürzerer Vorbau bringt dir mehr Kontrolle und entlastet den unteren Rücken.")
            elif s_angle < 75:
                st.warning("⚠️ **Lenker-Reach / Vorbau:** Du sitzt **zu gestaucht**.")
                st.write("*Empfehlung:* Du benötigst eventuell einen etwas längeren Vorbau oder musst den Sattel (falls möglich) nach hinten schieben (Setback kontrollieren).")
            else:
                st.success("🎉 **Lenker-Ergonomie:** Dein Schulterwinkel (80°-90°) ist ideal. Die Lastverteilung zwischen Händen und Sattel stimmt.")
                
        else:
            st.error("Es konnten im gesamten Video keine Gelenkpunkte verlässlich getrackt werden. Achte auf gute Ausleuchtung und ein kontrastreiches Profil von der Seite.")
