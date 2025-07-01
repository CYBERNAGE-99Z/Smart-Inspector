# 🛠️ Smart Inspector - AI-Powered Visual Quality Inspection

Smart Inspector is a computer vision project that uses **YOLOv8**, a state-of-the-art object detection model, to **automatically detect and classify defects** in industrial materials. It is trained on the **MVTec Anomaly Detection Dataset** and helps streamline quality control in manufacturing environments.

---

## 📌 Project Overview

In many industrial settings, quality inspection is still done manually, which is time-consuming and error-prone. Smart Inspector aims to automate this process by using deep learning to detect four key types of defects in metal nuts:

- 🔧 **Bent**
- 🎨 **Color anomaly**
- 🔄 **Flipped/misaligned**
- 🪵 **Scratched**

It also includes a web interface (built using **Streamlit**) that allows users to upload images and get instant defect detection results with bounding boxes and confidence scores.

---

## 📂 Dataset

The dataset used is from the official **MVTec AD (Anomaly Detection)** dataset:

- 📦 Class used: `metal_nut`
- ✅ Includes: Training images (`good`), test images (`good` + defects), and ground-truth segmentation masks for defects

🔗 **Dataset download**:  
👉 [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

## 📁 Folder Structure

```bash
Smart_Inspector/
├── best.pt                      # Trained YOLOv8 model
├── streamlit_app.py             # Web app (Streamlit interface)
├── main.ipynb  # Full training pipeline
├── datasets/
│   └── metal_nut_yolo/          # YOLO-compatible dataset
│       ├── images/train/
│       └── labels/train/

🚀 How It Works

    🔍 Preprocessing:

        Converts segmentation masks into YOLO-style bounding box annotations

        Organizes the dataset for YOLO training

    🧠 Model Training:

        Trains a YOLOv8 model (yolov8n.pt) on the labeled dataset

        Fine-tunes it to detect bent, color, flip, and scratch defects

    🧪 Prediction:

        Predicts defects on unseen test images using trained model

        Outputs annotated images with bounding boxes

    🌐 Web App:

        Users can upload images via the Streamlit app

        Get instant visual feedback with defect labels and confidence scores

⚙️ Technologies Used

    Python

    YOLOv8 (via ultralytics)

    OpenCV – image processing

    NumPy – data manipulation

    Streamlit – web interface

🧠 Credits

    Dataset: MVTec AD - Anomaly Detection

    YOLOv8: Ultralytics GitHub
