import glob
import os

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# =====================================================================
# SETTINGS - these MUST match the training notebook
# =====================================================================
IMG_SIZE = (224, 224)

# Class names in the SAME ORDER as the model's outputs (class 0, class 1, class 2...).
# For a 3-class model, replace None with your list, for example:
# Order used here = alphabetical order of the training folders (Keras default when class_names
# is not given): 0 = Abnormal heartbeat, 1 = History of MI, 2 = Normal Person.
# If your notebook set class_names in a different order, change this list to match.
CLASS_NAMES = ["Abnormal heartbeat", "History of MI", "Normal Person"]

# Two-class (single sigmoid output) models: Normal = 0, MI = 1
BINARY_NAMES = ["Normal", "History of MI"]
THRESHOLD = 0.5

# Did the training notebook crop the image (2% sides, 18% top, 5% bottom) before resizing?
CROP_LIKE_TRAINING = True

# Preferred model files, first match wins. Any other .keras file in this folder is used as a fallback.
PREFERRED = [
    "best_ecg_3class_cnn.keras",
    "ECG_Normal_vs_MI_CNN_v2.keras",
    "best_ecg_cnn_v2.keras",
    "ECG_Normal_vs_MI_CNN.keras",
    "ecg_normal_vs_mi_cnn.keras",
    "best_ecg_cnn.keras",
]

st.set_page_config(page_title="ECG AI Classifier", page_icon="🫀", layout="centered")
HERE = os.path.dirname(os.path.abspath(__file__))


def find_model_file():
    found = sorted(glob.glob(os.path.join(HERE, "*.keras")))
    names = {os.path.basename(p): p for p in found}
    for n in PREFERRED:
        if n in names:
            return names[n]
    for p in found:
        if "3class" in os.path.basename(p).lower():
            return p
    return found[0] if found else None


@st.cache_resource(show_spinner="Loading CNN model...")
def load_model():
    import tensorflow as tf

    path = find_model_file()
    if path is None:
        return None, None
    return tf.keras.models.load_model(path), os.path.basename(path)


def crop_ecg_region(img):
    w, h = img.size
    left, right = int(0.02 * w), int(0.98 * w)
    top, bottom = int(0.18 * h), int(0.95 * h)
    if bottom <= top or right <= left:
        return img
    return img.crop((left, top, right, bottom))


def preprocess(img):
    import tensorflow as tf

    img = ImageOps.exif_transpose(img).convert("RGB")
    used = crop_ecg_region(img) if CROP_LIKE_TRAINING else img
    # Same resize as training (tf.image.resize, bilinear)
    arr = tf.image.resize(np.asarray(used, dtype=np.float32), IMG_SIZE, method="bilinear").numpy() / 255.0
    return used, np.expand_dims(arr, axis=0)


def to_probabilities(raw):
    raw = np.asarray(raw, dtype=np.float64).ravel()
    if raw.size == 1:
        return raw  # single sigmoid output
    if raw.min() < 0 or raw.max() > 1 or abs(raw.sum() - 1) > 1e-3:
        e = np.exp(raw - raw.max())  # the model returned raw scores, turn into probabilities
        raw = e / e.sum()
    return raw


st.title("🫀 ECG AI Classifier")
st.caption("CNN-based ECG image classification")

model, model_name = load_model()
if model is None:
    st.error(
        "No model file found. Put your trained `.keras` model in the same folder as app.py "
        "in the GitHub repository."
    )
    st.stop()

n_out = int(model.output_shape[-1])
st.caption(f"Model file in use: {model_name}  |  outputs: {n_out}")

names = None
if n_out == 1:
    names = BINARY_NAMES
elif CLASS_NAMES is not None and len(CLASS_NAMES) == n_out:
    names = list(CLASS_NAMES)
else:
    names = [f"Class {i}" for i in range(n_out)]
    st.warning(
        f"The model has {n_out} classes, but CLASS_NAMES in app.py is not set (or has the wrong length). "
        "Results are shown as Class 0, Class 1, ... until you fill in the class names in the same order as training."
    )

uploaded = st.file_uploader("1. Upload an ECG image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded ECG", use_container_width=True)

    if st.button("2. Predict ECG Class", type="primary", use_container_width=True):
        used, batch = preprocess(image)
        with st.spinner("Analyzing..."):
            raw = model.predict(batch, verbose=0)[0]
        probs = to_probabilities(raw)

        st.subheader("3. Result")
        if n_out == 1:
            p_mi = float(probs[0])
            is_mi = p_mi >= THRESHOLD
            label = names[1] if is_mi else names[0]
            conf = p_mi if is_mi else 1 - p_mi
            (st.error if is_mi else st.success)(f"Predicted class: **{label}**")
            c1, c2 = st.columns(2)
            c1.metric("MI-class probability", f"{p_mi:.3f}")
            c2.metric("Model confidence", f"{conf * 100:.1f}%")
            st.progress(min(max(p_mi, 0.0), 1.0), text="MI-class probability")
        else:
            k = int(np.argmax(probs))
            st.success(f"Predicted class: **{names[k]}**")
            st.metric("Model confidence", f"{probs[k] * 100:.1f}%")
            for name, p in zip(names, probs):
                st.progress(float(min(max(p, 0.0), 1.0)), text=f"{name}: {p * 100:.1f}%")

        with st.expander("Image the model actually saw (cropped)"):
            st.image(used, use_container_width=True)

st.divider()
st.caption(
    "Academic prototype for a final-year project. Not clinically validated and "
    "not a tool for diagnosis or treatment."
)
