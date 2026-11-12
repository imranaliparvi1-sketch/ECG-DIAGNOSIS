import os

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ---- Settings that MUST match the training notebook ----
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "MI"]  # Normal = 0, MI = 1
THRESHOLD = 0.5
MODEL_CANDIDATES = [
    "ECG_Normal_vs_MI_CNN.keras",
    "ecg_normal_vs_mi_cnn.keras",
    "best_ecg_cnn.keras",
]

st.set_page_config(page_title="ECG AI Classifier", page_icon="🫀", layout="centered")


@st.cache_resource(show_spinner="Loading CNN model...")
def load_model():
    import tensorflow as tf

    for name in MODEL_CANDIDATES:
        if os.path.exists(name):
            return tf.keras.models.load_model(name), name
    return None, None


def crop_ecg_region(img):
    # Same crop as the notebook: 2% left/right, 18% top, 5% bottom.
    w, h = img.size
    left, right = int(0.02 * w), int(0.98 * w)
    top, bottom = int(0.18 * h), int(0.95 * h)
    if bottom <= top or right <= left:
        return img
    return img.crop((left, top, right, bottom))


def preprocess(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    cropped = crop_ecg_region(img)
    arr = np.asarray(cropped.resize(IMG_SIZE), dtype=np.float32) / 255.0
    return cropped, np.expand_dims(arr, axis=0)


st.title("🫀 ECG AI Classifier")
st.caption("CNN-based ECG image classification: Normal vs History of MI")

model, model_name = load_model()
if model is None:
    st.error(
        "Model file not found. Put your trained model (for example "
        "`ECG_Normal_vs_MI_CNN.keras`) in the same folder as app.py."
    )
    st.stop()

uploaded = st.file_uploader("1. Upload an ECG image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded ECG", use_container_width=True)

    if st.button("2. Predict ECG Class", type="primary", use_container_width=True):
        cropped, batch = preprocess(image)
        with st.spinner("Analyzing..."):
            prob_mi = float(model.predict(batch, verbose=0)[0][0])

        is_mi = prob_mi >= THRESHOLD
        label = "History of MI" if is_mi else "Normal"
        confidence = prob_mi if is_mi else 1 - prob_mi

        st.subheader("3. Result")
        if is_mi:
            st.error(f"Predicted class: **{label}**")
        else:
            st.success(f"Predicted class: **{label}**")

        c1, c2 = st.columns(2)
        c1.metric("MI-class probability", f"{prob_mi:.3f}")
        c2.metric("Model confidence", f"{confidence * 100:.1f}%")
        st.progress(min(max(prob_mi, 0.0), 1.0), text="MI-class probability")

        with st.expander("Image the model actually saw (cropped)"):
            st.image(cropped, use_container_width=True)

st.divider()
st.caption(
    "Academic prototype for a final-year project. Not clinically validated and "
    "not a tool for diagnosis or treatment."
)
