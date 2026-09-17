import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Malaria Detection AI",
    page_icon="🔬",
    layout="centered"
)

# 2. Load Model (with Caching for performance)
@st.cache_resource
def load_disease_model():
   
    return load_model('malaria_cnn_model.keras')

model = load_disease_model()

# 3. Sidebar (Professional touch)
st.sidebar.title("📌 About the App")
st.sidebar.info(
    "This AI-powered application uses a Convolutional Neural Network (CNN) "
    "to detect Malaria in microscopic images of red blood cells.\n\n"
    "**Accuracy:** ~95.79%"
)

# 4. Main UI
st.title("🔬 Malaria Cell Image Classifier")
st.write("Upload a microscopic image of a red blood cell to detect if it is **Parasitized** (infected) or **Uninfected**.")

# 5. File Uploader
uploaded_file = st.file_uploader("Choose a cell image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Cell Image', width=300)
    
    # Add a prediction button
    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing cell structure..."):
            
            # --- Image Preprocessing ---
            # 1. Resize to match training input (130x130)
            img_resized = image.resize((130, 130))
            
            # 2. Convert to numpy array
            img_array = np.array(img_resized)
            
            # 3. Scale pixel values (Normalization)
            img_array = img_array / 255.0
            
            # 4. Expand dimensions to create a batch of 1 -> shape: (1, 130, 130, 3)
            img_batch = np.expand_dims(img_array, axis=0)
            
            # --- Prediction ---
            prediction = model.predict(img_batch)[0][0]
            
            # --- Result Interpretation ---
            # Alphabetically: Parasitized is 0, Uninfected is 1
            if prediction < 0.5:
                # Parasitized
                confidence = (1 - prediction) * 100
                st.error(f"⚠️ **Result:** The cell is **Parasitized** (Infected with Malaria).")
                st.write(f"**Confidence:** {confidence:.2f}%")
            else:
                # Uninfected
                confidence = prediction * 100
                st.success(f"✅ **Result:** The cell is **Uninfected** (Healthy).")
                st.write(f"**Confidence:** {confidence:.2f}%")
                
            st.progress(int(confidence))