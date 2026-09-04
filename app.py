
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #aaaaaa;
    margin-bottom: 30px;
}

.result-box {
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
    font-size: 20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.markdown(
    '<div class="title">😷 Face Mask Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered face mask detection using Computer Vision</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

MODEL_PATH = "models/mask_detector.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

# --------------------------------------------------
# Face Detector
# --------------------------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def detect_mask(image):

    # Convert PIL image to OpenCV
    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    results = []

    for (x, y, w, h) in faces:

        face = img[y:y+h, x:x+w]

        if face.size == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face = cv2.resize(face, (224, 224))

        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=0)

        prediction = model.predict(
            face,
            verbose=0
        )[0][0]

        # Model convention:
        # 0 = With Mask
        # 1 = Without Mask

        if prediction < 0.5:
            label = "Mask"
            confidence = (1 - prediction) * 100
        else:
            label = "No Mask"
            confidence = prediction * 100

        results.append(
            (x, y, w, h, label, confidence)
        )

    return img, results


# --------------------------------------------------
# Input Method
# --------------------------------------------------

st.subheader("Choose input method:")

input_method = st.radio(
    "",
    ["Upload Photo", "Use Camera"],
    horizontal=False
)

# --------------------------------------------------
# Upload Photo
# --------------------------------------------------

image = None

if input_method == "Upload Photo":

    uploaded_file = st.file_uploader(
        "Upload a photo",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

# --------------------------------------------------
# Camera
# --------------------------------------------------

else:

    camera_image = st.camera_input(
        "Take a picture"
    )

    if camera_image is not None:

        image = Image.open(camera_image)

# --------------------------------------------------
# Detection
# --------------------------------------------------

if image is not None:

    st.subheader("Detection Results")

    if model is None:

        st.error(
            "Model not found. Please put "
            "`mask_detector.keras` inside the models folder."
        )

    else:

        output, results = detect_mask(image)

        # Convert BGR → RGB
        output = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        )

        # Draw detection boxes
        for (x, y, w, h, label, confidence) in results:

            if label == "Mask":
                box_color = (0, 255, 0)
            else:
                box_color = (255, 0, 0)

            cv2.rectangle(
                output,
                (x, y),
                (x+w, y+h),
                box_color,
                3
            )

            text = f"{label} ({confidence:.2f}%)"

            cv2.putText(
                output,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )

        st.image(
            output,
            caption="Face Mask Detection",
            use_container_width=True
        )

        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        if len(results) == 0:

            st.warning(
                "No face detected in the image."
            )

        else:

            for i, result in enumerate(results):

                x, y, w, h, label, confidence = result

                if label == "Mask":

                    st.success(
                        f"Face {i+1}: 😷 Mask detected "
                        f"({confidence:.2f}%)"
                    )

                else:

                    st.error(
                        f"Face {i+1}: ⚠️ No Mask detected "
                        f"({confidence:.2f}%)"
                    )


# --------------------------------------------------
# Clear History
# --------------------------------------------------

st.divider()

if st.button("🗑️ Clear History"):

    st.rerun()

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Face Mask Detector | Python • TensorFlow • OpenCV • Streamlit"
)
