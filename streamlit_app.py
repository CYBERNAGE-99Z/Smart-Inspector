import streamlit as st
from ultralytics import YOLO
import os
import shutil
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from io import BytesIO

# Title
st.set_page_config(page_title="Smart Inspector", layout="centered")
st.title("🛠️ Smart Inspector - AI-Powered Visual Quality Check")
st.caption("Upload an image of a metal nut to detect visual defects using YOLOv8")

# Load the trained YOLOv8 model
model = YOLO("best.pt")

# Create directories for upload and output if they don't exist
upload_dir = "uploaded_images"
output_dir = "detection_results"
os.makedirs(upload_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Confidence threshold
conf_threshold = st.slider("🎯 Set Detection Confidence Threshold", 0.1, 1.0, 0.25)

# Image uploader
uploaded_file = st.file_uploader("📤 Upload a test image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Clear previous uploads and results
    shutil.rmtree(upload_dir)
    shutil.rmtree(output_dir)
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Save uploaded image
    image_path = os.path.join(upload_dir, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run detection
    results = model.predict(source=image_path, conf=conf_threshold, save=True, save_txt=False, project=output_dir, name="predict", exist_ok=True)
    # Dynamically find the saved image (YOLO sometimes renames it)
    output_subdir = os.path.join(output_dir, "predict")
    saved_files = [f for f in os.listdir(output_subdir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if saved_files:
        result_image_path = os.path.join(output_subdir, saved_files[0])
    else:
        st.warning("⚠️ Detection image not found. Something went wrong during prediction.")


    # Display side-by-side images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_path, caption="📷 Original Image", use_column_width=True)
    with col2:
        st.image(result_image_path, caption="🎯 Detected Defects", use_column_width=True)

    # Detection summary
    st.markdown("## 🧾 Detection Summary")
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        st.success("✅ No defects detected. This item passes quality inspection.")
    else:
        st.error(f"❌ {len(boxes)} defect(s) found. Item flagged for review.")
        defect_data = []
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            st.write(f"- **{label}** with confidence **{confidence:.2f}**")
            defect_data.append({"Defect Type": label, "Confidence": f"{confidence:.2f}"})

        # Show table
        df = pd.DataFrame(defect_data)
        st.table(df)

        # CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Detection Report (CSV)", csv, "defect_report.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [YOLOv8](https://github.com/ultralytics/ultralytics) and [Streamlit](https://streamlit.io/)")
