import os
import pickle
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------- SETTINGS ----------------
BASE_DIR = "Main_py"

# ---------------- FILE CHECK ----------------
def check_file(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        st.error(f"❌ File '{filename}' not found in '{BASE_DIR}'")
        st.stop()
    return path

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_all_models():
    extractor = load_model(check_file("feature_extractor.keras"))
    with open(check_file("svm_model.pkl"), "rb") as f:
        svm = pickle.load(f)
    with open(check_file("rf_model.pkl"), "rb") as f:
        rf = pickle.load(f)
    with open(check_file("xgb_model.pkl"), "rb") as f:
        xgb = pickle.load(f)
    return extractor, svm, rf, xgb

feature_extractor, svm_model, rf_model, xgb_model = load_all_models()

# Face detector for rejecting non-face inputs
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------------- PREDICTION FUNCTION ----------------
def predict_single_image(uploaded_file):
    # Reset file pointer and read bytes
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        # Could not decode image
        blank = np.zeros((300, 300, 3), dtype=np.uint8)
        return None, blank, 0.0

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_display = cv2.resize(img_rgb, (300, 300), interpolation=cv2.INTER_AREA)

    # --- Face detection ---
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None, img_display, 0.0  # Reject if no face detected

    # --- Preprocess for model ---
    img_model = cv2.resize(img_rgb, (128, 128))
    img_model = cv2.cvtColor(img_model, cv2.COLOR_RGB2GRAY) / 255.0
    img_model = np.expand_dims(img_model, axis=-1)  # (128, 128, 1)
    img_model = np.expand_dims(img_model, axis=0)   # (1, 128, 128, 1)

    # --- Extract features ---
    feature_vector = feature_extractor.predict(img_model)
    feature_vector = feature_vector.reshape(1, -1)

    # --- Get probabilities from models ---
    def safe_proba(model):
        try:
            return model.predict_proba(feature_vector)[0]
        except:
            pred = model.predict(feature_vector)[0]
            return [1.0 - pred, pred] if pred in [0, 1] else [0.5, 0.5]

    prob_svm = safe_proba(svm_model)
    prob_rf = safe_proba(rf_model)
    prob_xgb = safe_proba(xgb_model)

    # Average probabilities
    avg_probs = np.mean([prob_svm, prob_rf, prob_xgb], axis=0)
    confidence = float(np.max(avg_probs))
    predicted_class = int(np.argmax(avg_probs))
    gender_final = "Male" if predicted_class == 1 else "Female"

    # --- Reject if too low confidence ---
    if confidence < 0.70:
        return None, img_display, confidence

    return gender_final, img_display, confidence

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Gender Classification App", page_icon="👤")

st.title("👤 Gender Classification (Hybrid DL + ML)")
st.write("Only predicts for **clear human male/female faces** — others will be rejected but still shown.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.button("🔍 Predict Gender"):
        gender, img_display, conf = predict_single_image(uploaded_file)

        fig, ax = plt.subplots()
        ax.imshow(img_display)
        ax.axis("off")

        if gender is None:
            msg = "❌ This is not detected as a picture of a man or woman."
            ax.set_title(msg, fontsize=10, color="red")
            st.pyplot(fig)
            st.warning(msg)
        else:
            color = "blue" if gender == "Male" else "green"
            ax.set_title(f"{gender} ({conf*100:.1f}% confident)", fontsize=12, color=color)
            st.pyplot(fig)
            st.success(f"✅ Predicted Gender: {gender} — {conf*100:.1f}% confidence")
