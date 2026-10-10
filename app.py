"""
Master plan Classification - Streamlit web app
--------------------------------------
Users upload image files; the trained CNN classifies each one as
VALID or INVALID, shows the confidence (%), and lets them download
ALL the valid images as a single ZIP.

The trained model is NOT stored in this repo. It is downloaded once
from a Hugging Face model repository at startup and cached.

Run locally:   streamlit run app.py
"""

import io
import zipfile

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

# ------------------------------------------------------------------
# Must match the settings used during TRAINING
# ------------------------------------------------------------------
IMG_SIZE = (128, 128)      # same as training
CHANNELS = 3               # 3 = RGB (set to 1 if you trained on grayscale)

# ------------------------------------------------------------------
# Where the model lives on Hugging Face  (EDIT THESE TWO LINES)
# ------------------------------------------------------------------
HF_REPO_ID = "ahmedkindi/mp-classification"   # <-- your HF repo id
HF_FILENAME = "cnn_mp_classification.keras"      # <-- the file name in that repo

# Label mapping used in training: invalid -> 0, valid -> 1
CLASS_NAMES = {0: "invalid", 1: "valid"}

# Threshold for the sigmoid output
THRESHOLD = 0.7

# ------------------------------------------------------------------
# Download the model from Hugging Face (once) and cache it
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    # hf_hub_download caches the file, so it only downloads the first time
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    return tf.keras.models.load_model(model_path)

with st.spinner("Loading model… (first run downloads it from Hugging Face)"):
    model = load_model()

# ------------------------------------------------------------------
# Preprocess a single uploaded image exactly like in training
# ------------------------------------------------------------------
def preprocess(pil_image):
    mode = "RGB" if CHANNELS == 3 else "L"
    img = pil_image.convert(mode).resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32") / 255.0      # normalize to [0,1]
    if CHANNELS == 1:
        arr = np.expand_dims(arr, axis=-1)              # add channel dim
    return np.expand_dims(arr, axis=0)                  # add batch dim

# ------------------------------------------------------------------
# Turn a raw sigmoid output into (status, confidence %)
# ------------------------------------------------------------------
def interpret(prob):
    if prob >= THRESHOLD:
        status = CLASS_NAMES[1]          # valid
        confidence = prob * 100
    else:
        status = CLASS_NAMES[0]          # invalid
        confidence = (1 - prob) * 100
    return status, confidence

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Master Plan  Classifier", page_icon="🔎", layout="wide")
st.title("🔎 Master Plan Image Classification")
st.write(
    "Upload one or more images (select every image in your folder). "
    "Each image is classified as **valid** or **invalid** with a confidence score. "
    "Afterwards you can download all the valid images as a single ZIP."
)

uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.subheader("Results")
    results = []
    valid_files = []        # (filename, raw_bytes) -> goes into the ZIP

    for f in uploaded_files:
        raw_bytes = f.getvalue()        # keep original bytes for the ZIP
        try:
            pil_image = Image.open(io.BytesIO(raw_bytes))
            batch = preprocess(pil_image)
            prob = float(model.predict(batch, verbose=0)[0][0])
            status, confidence = interpret(prob)

            if status == "valid":
                valid_files.append((f.name, raw_bytes))

            results.append(
                {
                    "Image name": f.name,
                    "Verification status": status,
                    "Confidence (%)": f"{confidence:.f}",
                }
            )

            # Per-image display with thumbnail
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(pil_image, width=120)
            with col2:
                st.markdown(f"**Image name:** {f.name}")
                if status == "valid":
                    st.success(f"Verification status: **{status}**")
                else:
                    st.error(f"Verification status: **{status}**")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
            st.divider()

        except Exception as e:
            st.warning(f"Could not process {f.name}: {e}")

    # --------------------------------------------------------------
    # Summary table
    # --------------------------------------------------------------
    if results:
        st.subheader("Summary table")
        st.table(results)

        n_valid = len(valid_files)
        n_total = len(results)
        st.info(f"{n_valid} of {n_total} images classified as VALID.")

        # ----------------------------------------------------------
        # Download link: ONLY the valid images, as a ZIP
        # ----------------------------------------------------------
        if valid_files:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, data in valid_files:
                    zf.writestr(name, data)
            zip_buffer.seek(0)

            st.download_button(
                "⬇️ Download valid images (ZIP)",
                data=zip_buffer,
                file_name="valid_images.zip",
                mime="application/zip",
            )
        else:
            st.info("No valid images to download.")

        # ----------------------------------------------------------
        # Also offer the full results table as CSV
        # ----------------------------------------------------------
        import csv
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        st.download_button(
            "Download results as CSV",
            data=buf.getvalue(),
            file_name="classification_results.csv",
            mime="text/csv",
        )
else:
    st.info("Waiting for images… use the uploader above.")
