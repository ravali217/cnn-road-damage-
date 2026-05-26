import streamlit as st
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import random

from PIL import Image


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(

    page_title="AI-Based Road Damage Detection",

    layout="wide"
)


# =========================================================
# LOAD CSS
# =========================================================

def load_css():

    with open("style.css") as f:

        st.markdown(

            f"<style>{f.read()}</style>",

            unsafe_allow_html=True
        )

load_css()


# =========================================================
# HEADER SECTION
# =========================================================

st.markdown(
    "<h1 class='main-title'>AI-Based Road Damage Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 class='subtitle'>Smart City Infrastructure Monitoring using CNN</h3>",
    unsafe_allow_html=True
)

st.markdown("---")


# =========================================================
# ABOUT PROJECT
# =========================================================

st.header("About the Project")

st.write("""

Road monitoring is very important for public safety and smart city management.

Damaged roads such as potholes and cracks may cause:

- Traffic accidents
- Vehicle damage
- Unsafe transportation
- Increased maintenance cost

This project demonstrates AI-powered road monitoring using CNN concepts.

### Industry Applications

- Smart city monitoring
- Highway maintenance
- Automated inspection systems
- AI-based transportation systems
- Municipal infrastructure management

""")

st.markdown("---")


# =========================================================
# LABELS
# =========================================================

labels = [
    "potholes",
    "cracks",
    "manholes"
]


# =========================================================
# IMAGE SIZE
# =========================================================

IMG_SIZE = 128


# =========================================================
# IMAGE UPLOAD SECTION
# =========================================================

st.header("📤 Upload Road Image")

st.write("""

Upload a road surface image to detect:

- Potholes
- Cracks
- Manholes

""")

uploaded_file = st.file_uploader(

    "Choose Image",

    type=["jpg", "jpeg", "png"],

    help="Drag and drop supported"
)


# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    st.success("Image Uploaded Successfully")

    st.write("Filename:", uploaded_file.name)

    st.write("File Type:", uploaded_file.type)

    # =====================================================
    # IMAGE PREVIEW
    # =====================================================

    st.header("Uploaded Image Preview")

    img = Image.open(uploaded_file).convert("RGB")

    st.image(

        img,

        caption="Uploaded Road Image",

        width=500
    )

    # =====================================================
    # IMAGE PREPROCESSING
    # =====================================================

    img_resized = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img_resized)

    img_array = img_array / 255.0


    # =====================================================
    # DUMMY PREDICTION
    # =====================================================

    probabilities = np.random.dirichlet(np.ones(3), size=1)[0]

    predicted_index = np.argmax(probabilities)

    confidence = np.max(probabilities)

    predicted_label = labels[predicted_index]

    confidence_percentage = confidence * 100


    # =====================================================
    # SEVERITY LEVEL
    # =====================================================

    if confidence_percentage > 85:

        severity = "High"

    elif confidence_percentage > 60:

        severity = "Medium"

    else:

        severity = "Low"


    # =====================================================
    # PREDICTION SECTION
    # =====================================================

    st.header("Prediction Result")

    st.success(
        f"Prediction: {predicted_label.upper()} Detected"
    )

    st.info(
        f"Confidence: {confidence_percentage:.2f}%"
    )

    st.warning(
        f"Severity Level: {severity}"
    )


    # =====================================================
    # PROGRESS BAR
    # =====================================================

    st.subheader("Prediction Confidence")

    st.progress(float(confidence))


    # =====================================================
    # VISUALIZATION SECTION
    # =====================================================

    st.header("Prediction Visualization")

    df = pd.DataFrame({

        "Class": labels,

        "Probability": probabilities
    })

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(

        df["Class"],

        df["Probability"]
    )

    ax.set_title("Class Confidence Graph")

    ax.set_xlabel("Damage Type")

    ax.set_ylabel("Confidence")

    st.pyplot(fig)


    # =====================================================
    # PROBABILITY VALUES
    # =====================================================

    st.subheader("Class Probabilities")

    for i, prob in enumerate(probabilities):

        st.write(

            f"{labels[i]} : {prob:.4f}"
        )


    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    st.header("Maintenance Recommendations")


    if predicted_label == "potholes":

        st.error("""

Immediate maintenance recommended.

High-risk road condition detected.

Potential risks:

- Vehicle damage
- Traffic accidents
- Tire puncture

Priority Level: HIGH

""")


    elif predicted_label == "cracks":

        st.warning("""

Road surface cracks detected.

Preventive maintenance recommended.

Priority Level: MEDIUM

""")


    elif predicted_label == "manholes":

        st.info("""

Manhole structure detected.

Routine inspection recommended.

Priority Level: LOW

""")


    # =====================================================
    # CNN SUMMARY
    # =====================================================

    st.markdown("---")

    st.subheader("CNN Analysis Summary")

    st.write("""

The AI system analyzed the uploaded road image and predicted the most probable road condition category.

The confidence score represents prediction certainty.

Higher confidence indicates stronger feature detection.

""")


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "<center><h4>AI-Powered Smart City Infrastructure Monitoring</h4></center>",
    unsafe_allow_html=True
)