import streamlit as st
from PIL import Image
import os
import uuid
from ultralytics import YOLO

# -------------------------------
# APP CONFIG
# -------------------------------
st.set_page_config(
    page_title="Waste Segmentation using YOLO",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f3f7fa;
    }
    .main {
        background-color: white;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #1a1a1a;
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #007bff, #00c6ff);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #0056b3, #00aaff);
    }
    .css-1v0mbdj img {
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# DIRECTORIES
# -------------------------------
UPLOAD_DIR = os.path.join("predicts", "uploaded_images")
OUTPUT_DIR = os.path.join("predicts", "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO("best.pt")  # Ensure best.pt is in the same folder

# -------------------------------
# APP HEADER
# -------------------------------
st.markdown("<h1>♻️ Waste Segmentation using YOLOv8</h1>", unsafe_allow_html=True)
st.write("Upload an image to visualize segmentation and predicted waste types.")

# -------------------------------
# UPLOAD SECTION
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Save uploaded image
    file_ext = uploaded_file.name.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    uploaded_path = os.path.join(UPLOAD_DIR, unique_filename)
    image.save(uploaded_path)

    # Run YOLOv8 segmentation
    with st.spinner("🧠 Running YOLOv8 segmentation..."):
        results = model(uploaded_path, project="predicts", name="output", save=True, exist_ok=True)

    # Extract detected class names
    classes = results[0].names
    labels = set([classes[int(cls)] for cls in results[0].boxes.cls])

    # Find annotated image
    annotated_path = os.path.join("predicts", "output", os.path.basename(uploaded_path))

    # -------------------------------
    # DISPLAY SIDE-BY-SIDE IMAGES
    # -------------------------------
    st.markdown("---")
    st.subheader("🔍 Segmentation Result")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🖼 Uploaded Image")
        st.image(image, caption="Original Waste Image", use_container_width=True)

    with col2:
        st.markdown("#### ♻️ Segmented Output")
        if os.path.exists(annotated_path):
            annotated_img = Image.open(annotated_path)
            st.image(annotated_img, caption="YOLOv8 Segmentation Result", use_container_width=True)
        else:
            st.warning("⚠️ Annotated image not found.")

    # -------------------------------
    # PREDICTION RESULTS
    # -------------------------------
    st.markdown("---")
    st.markdown("### 🧠 Model Prediction")
    st.success("**Detected Waste Type(s):** " + ", ".join(labels))

    st.markdown("""
    <div style='text-align:center; margin-top:30px; color:#666;'>
        <small>Developed using <b>YOLOv8</b> & <b>Streamlit</b> | Powered by Abhilash & Tazrian</small>
    </div>
    """, unsafe_allow_html=True)
