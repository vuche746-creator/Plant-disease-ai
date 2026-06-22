
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
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)
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

    image = image.resize((224,224))

    img_array = np.array(image)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction)) * 100

    predicted_class = class_names[predicted_index]

    parts = predicted_class.split("___")

    plant_name = parts[0].replace("_", " ")

    if len(parts) > 1:
        disease_name = parts[1].replace("_", " ")
    else:
        disease_name = "Unknown"

    info = disease_info.get(predicted_class, {})

    description = info.get(
        "description",
        "No description available."
    )

    treatment = info.get(
        "treatment",
        ["No treatment information available."]
    )

    st.success("Prediction Complete")

    st.subheader("🌱 Plant")
    st.write(plant_name)

    if "healthy" in predicted_class.lower():
        st.subheader("✅ Status")
        st.write("Healthy")
    else:
        st.subheader("🦠 Disease")
        st.write(disease_name)

    st.subheader("📊 Confidence")
    st.write(f"{confidence:.2f}%")

    st.subheader("📖 Description")
    st.write(description)

    st.subheader("💊 Recommended Treatment")

    for item in treatment:
        st.write(f"• {item}")
