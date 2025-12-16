import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

# 1. Load the trained model using pickle
@st.cache_resource
def load_model():
    try:
        # Load the model from the pickle file
        with open('Alzhimer.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# 2. Define Class Names
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# 3. App Title and Description
st.set_page_config(page_title="Alzheimer's Detection", page_icon="🧠")
st.title("🧠 Alzheimer's Disease Detection")
st.write("Upload an MRI image to detect the stage of Alzheimer's.")

# 4. File Uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI', use_column_width=True)
    
    st.write("Analyzing...")

    # 5. Preprocessing
    try:
        # Resize to 224x224
        img = image.resize((224, 224))
        
        # Convert to numpy array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Expand dims to make it (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize
        img_array = img_array / 225.0
        
        # 6. Make Prediction
        if model:
            prediction = model.predict(img_array)
            ind = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            result_class = class_names[ind]
            
            # 7. Display Result
            st.markdown("---")
            st.subheader("Prediction Result:")
            
            if result_class == "NonDemented":
                st.success(f"**{result_class}** (Confidence: {confidence*100:.2f}%)")
            else:
                st.warning(f"**{result_class}** (Confidence: {confidence*100:.2f}%)")
                
    except Exception as e:
        st.error(f"Error processing image: {e}")
