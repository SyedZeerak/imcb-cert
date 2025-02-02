import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw
import time
import cv2

# Custom CSS for a modern Neon Green UI with improved design
def add_custom_css():
    st.markdown(
        """
        <style>
            body {background-color: #0a0f0d; font-family: 'Arial', sans-serif; color: #39ff14;}
            h1 {color: #39ff14; text-align: center; font-size: 2.5em; font-weight: bold;}
            .stButton>button {background-color: #39ff14; color: black; font-size: 1.1em; padding: 10px 20px; border-radius: 10px; transition: 0.3s;}
            .stButton>button:hover {background-color: #32cd32; transform: scale(1.05);}
            .metric-box {background: #1a2a1a; border-radius: 12px; padding: 15px; box-shadow: 0 5px 12px rgba(0,255,0,0.3); text-align: center; border: 1px solid #39ff14;}
            .metric-box h4 {color: #39ff14; font-size: 1.2em; margin-bottom: 5px;}
            .metric-box p {font-size: 1.5em; font-weight: bold;}
        </style>
        """,
        unsafe_allow_html=True,
    )

add_custom_css()

# Load YOLO model
model = YOLO("X:\\2PersonalProjects\\04Computer VIsion\\imcb-certs\\model.pt")
CLASS_NAMES = ["certificate", "logo", "title"]

RESULT_CATEGORIES = {
    "valid": "IMCB Certificate ✅",
    "generic": "Generic Certificate ⚠️",
    "other": "Other Object ❌"
}

def categorize_certificate(detection):
    if detection["cert_conf"] >= 0.65 and (detection["has_logo"] or detection["has_title"]):
        return RESULT_CATEGORIES["valid"], "#39ff14"
    elif detection["cert_conf"] >= 0.65:
        return RESULT_CATEGORIES["generic"], "#f1c40f"
    else:
        return RESULT_CATEGORIES["other"], "#e74c3c"

def draw_bounding_boxes(image, results):
    draw = ImageDraw.Draw(image)
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            class_name = CLASS_NAMES[int(cls)]
            box = list(map(int, box.cpu().numpy()))
            draw.rectangle(box, outline="#39ff14", width=3)
            draw.text((box[0], box[1] - 10), f"{class_name}: {conf:.2f}", fill="#39ff14")
    return image

def process_detection(results):
    detections = {"certificate": [], "logo": [], "title": []}
    
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            class_name = CLASS_NAMES[int(cls)]
            detections[class_name].append({"box": box.cpu().numpy().tolist(), "confidence": float(conf)})
    
    best_cert = max(detections["certificate"], key=lambda x: x["confidence"], default=None)
    
    if not best_cert:
        return None
    
    return {
        "cert_conf": best_cert["confidence"],
        "has_logo": len(detections["logo"]) > 0,
        "has_title": len(detections["title"]) > 0,
        "box": best_cert["box"]
    }

st.image("images.jpg", width=150)
st.title("🎓 IMCB Certificate Validator")
st.write("Upload a document to validate IMCB F-8/4 certificates")

uploaded_file = st.file_uploader("Upload Certificate Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    with st.spinner("Analyzing document..."):
        time.sleep(2)
        results = model(image_np)
        detection = process_detection(results)
        image = draw_bounding_boxes(image, results)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Detected Certificate")
        st.image(image)
    
    with col2:
        if not detection:
            st.error("🚫 No certificate detected")
        else:
            result_text, result_color = categorize_certificate(detection)
            with st.expander("📌 View Result", expanded=True):
                st.markdown(
                    f"""
                    <div class='metric-box' style='border: 2px solid {result_color};'>
                        <h4>Final Result</h4>
                        <p style='color: {result_color}; font-size: 1.8em;'>{result_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #39ff14; font-size: 0.9em;">
        <strong>IMCB Certificate Validation System</strong> • Developed by <a href='https://github.com/SyedZeerak/' style='color:#39ff14;' target='_blank'>Syed Wajdan Zeerak</a>
    </div>
    """,
    unsafe_allow_html=True
)
