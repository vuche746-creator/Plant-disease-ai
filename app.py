
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model("plant_disease_final.h5")

# Load classes
with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("🌱 Plant Disease Detection AI")

uploaded_file = st.file_uploader(
    "Upload a Plant Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image = image.resize((224, 224))

    img_array = np.array(image)

    # Handle RGBA images
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction)

    disease = class_names[predicted_index]

    st.success(f"Disease: {disease}")
    st.info(f"Confidence: {confidence*100:.2f}%")
