import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import uuid

# Load trained YOLOv8 model
model = YOLO("best.pt")

st.title("🔍 Smart Inspector")
st.write("Upload an image of a metal nut or any other industrial material to detect various defects like **bent**, **color**, **flip**, and **scratch**.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the image temporarily
    image_id = str(uuid.uuid4())
    temp_path = f"temp_{image_id}.png"
    image.save(temp_path)

    # Predict using YOLO
    with st.spinner("Running detection..."):
        results = model.predict(source=temp_path, conf=0.25)
    
    result = results[0]
    boxes = result.boxes
    annotated_image = result.plot()  # BGR numpy array

    # Convert to RGB and show
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    st.image(annotated_image_rgb, caption="Detected Defects", use_column_width=True)

    # Show class names and confidences
    st.subheader("📋 Detection Details")
    class_names = model.names
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"**{class_names[cls_id]}** – Confidence: `{conf:.2f}`")

    os.remove(temp_path)  # Clean up temp file
