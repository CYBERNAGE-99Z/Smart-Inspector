# 🕵️‍♂️ Smart Inspector – AI-Powered Defect Detection in Manufacturing

Smart Inspector is a deep learning-based quality inspection tool designed to automatically detect manufacturing defects in real-time using computer vision. Built with YOLOv8 and Streamlit, it streamlines quality control by identifying issues such as **bent parts, color mismatches, scratches, and flipped components** in metal nuts on the assembly line.

---

## 🔍 Features

- ✅ **Multi-class Defect Detection** (bent, scratch, color, flip, good)
- 🚀 Real-time image inference via **Streamlit Web App**
- 🎯 Powered by **YOLOv8** object detection model
- 📈 Clear prediction confidence and visual bounding boxes
- 🧠 Easy retraining with custom datasets

---

## 📁 Folder Structure

Smart-Inspector/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│ ├── train/
│ ├── test/
│ └── ground_truth/
│ ├── bent/
│ ├── color/
│ ├── flip/
│ └── scratch/
│
├── yolov8_model/
│ ├── trained_model.pt
│ └── runs/
│
├── app/
│ ├── streamlit_app.py
│ └── utils.py
│
├── scripts/
│ ├── train.py
│ └── detect.py
│
└── notebooks/
└── EDA_and_Preprocessing.ipynb


---

## 🛠️ Installation

### 🔗 Clone the Repository

```bash
git clone https://github.com/your-username/Smart-Inspector.git
cd Smart-Inspector
