import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# 🌿 Set page config
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")

# 📦 Load model once
@st.cache_resource
def load_plant_model():
    return load_model("plant_disease_resnet152v2.keras")

model = load_plant_model()

# 🔖 Class Labels from PlantVillage dataset (38 classes)
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# 📚 Sidebar Navigation
page = st.sidebar.selectbox("📚 Navigate", ["About Project", "Predict Disease"])

# -------------------------------
# 📄 About Project Page
# -------------------------------
if page == "About Project":
    st.title("🌿 Plant Disease Detection Project")
    st.markdown("""
    ### 🧩 Problem Statement
    Farmers often struggle to identify crop diseases in time, leading to reduced yields and poor crop health. Manual diagnosis is slow and inaccurate. We aim to solve this using Deep Learning.

    ### 🧠 How It Works
    - **Dataset**: [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) with 38 classes
    - **Preprocessing**: Image augmentation, resized to 224x224
    - **Model**: Fine-tuned ResNet152V2
    - **Frameworks**: TensorFlow, Streamlit
    """)

    st.subheader("🌾 Sample Disease Images")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("images/Tomato_Leaf_Mold.jpeg", use_container_width=True)
        st.caption("Tomato Leaf Mold")
    with col2:
        st.image("images/Potato_Late_Blight.jpeg", use_container_width=True)
        st.caption("Potato Late Blight")
    with col3:
        st.image("images/Bell_Pepper_Bacterial_Spot.jpeg", use_container_width=True)
        st.caption("Bell Pepper Bacterial Spot")

    st.markdown("""
    ### 🚀 Future Enhancements
    - Integrate with **LLMs** for:
      - Detailed disease explanation
      - Pesticide suggestions
      - Region-based alerts
    - Mobile App Deployment

    👩‍💻 Made by **Varsha** | B.Tech Data Science
    """)

# -------------------------------
# 🔍 Disease Prediction Page
# -------------------------------
elif page == "Predict Disease":
    st.title("🔍 Predict Plant Disease")
    st.write("Upload a **leaf image** to detect the disease.")

    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        if st.button("🔬 Predict Now"):
            with st.spinner("Analyzing... Please wait..."):
                img = image.resize((224, 224))
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model.predict(img_array)[0]
                top2 = predictions.argsort()[-2:][::-1]
                confidence = predictions[top2[0]] * 100
                gap = predictions[top2[0]] - predictions[top2[1]]
                std_dev = np.std(predictions)
                predicted_class = class_labels[top2[0]]

                # 🚫 Reject unlikely predictions (OOD Detection)
                if confidence < 85 or gap < 0.5 or std_dev < 0.05:
                    st.error("❗ This image doesn't confidently match any known disease class.")
                    st.info(f"🔍 Top Guess: {class_labels[top2[0]]} ({predictions[top2[0]]:.2f})\n🧐 Second: {class_labels[top2[1]]} ({predictions[top2[1]]:.2f})")
                else:
                    st.success(f"🌿 **Prediction**: {predicted_class}")
                    st.info(f"🔬 **Confidence**: `{confidence:.2f}%`")

                    # 🧪 Optional Tips
                    if "blight" in predicted_class.lower():
                        st.warning("💡 Tip: Remove infected leaves, use copper-based fungicides.")
                    elif "spot" in predicted_class.lower():
                        st.warning("💡 Tip: Use resistant varieties, avoid overhead watering.")
                    elif "rust" in predicted_class.lower():
                        st.warning("💡 Tip: Apply sulfur-based fungicide and prune infected leaves.")
