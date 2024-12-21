# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
from gtts import gTTS
from googletrans import Translator
import playsound
import os
import tempfile

# Text-to-speech function for different languages
def speak_warning(message, language_code='en'):
    try:
        # Create a gTTS object
        tts = gTTS(text=message, lang=language_code)
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            audio_file = tmp_file.name
            tts.save(audio_file)

        # Play the audio file
        playsound.playsound(audio_file)

        # Remove the audio file after playing
        os.remove(audio_file)

    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")

# Function to generate and open URL based on disease
def open_disease_info(plant_type, disease):
    base_url = "https://en.wikipedia.org/wiki/"
    disease_url = f"{base_url}{disease}".replace(' ', '-')
    
    st.markdown(f"<a href='{disease_url}' target='_blank'>Click here to learn more about {plant_type} and {disease}</a>", unsafe_allow_html=True)

# Loading the Model
model = load_model('model/plant_disease_model_sanketh_mini.h5')

# Name of Classes
CLASS_NAMES = ['Apple_Apple scab', 'Apple_Black rot', 'Apple_Cedar rust', 'Apple_healthy', 
               'Potato_Early blight', 'Potato_healthy', 'Potato_Late blight', 
               'Tomato_Bacterial spot', 'Tomato_Early blight', 'Tomato_healthy']

# Define treatment and prevention messages for each disease
# Treatment and prevention messages for each disease
treatment_info = {
    'Apple_Apple scab': {
        'treatment': "Treatment includes removing affected leaves, applying fungicides, and ensuring proper airflow.",
        'prevention': "Prevention includes planting disease-resistant varieties and ensuring proper pruning."
    },
    'Apple_Black rot': {
        'treatment': "Treat with fungicides and ensure proper sanitation in orchards.",
        'prevention': "Prevent by removing infected plant debris and avoiding overhead watering."
    },
    'Apple_Cedar rust': {
        'treatment': "Prune affected parts and use fungicides as a preventive measure.",
        'prevention': "Prevent by removing nearby cedar trees and planting resistant apple varieties."
    },
    'Potato_Early blight': {
        'treatment': "Use fungicides and practice crop rotation to manage this disease.",
        'prevention': "Prevention includes avoiding dense planting and ensuring proper air circulation."
    },
    'Potato_Late blight': {
        'treatment': "Apply fungicides promptly and ensure good drainage in fields.",
        'prevention': "Prevention includes using resistant varieties and removing infected plant debris."
    },
    'Tomato_Bacterial spot': {
        'treatment': "Remove affected plants, use resistant varieties, and apply bactericides.",
        'prevention': "Prevention includes proper crop rotation and avoiding overhead watering."
    },
    'Tomato_Early blight': {
        'treatment': "Use resistant varieties, rotate crops, and apply fungicides.",
        'prevention': "Prevention includes ensuring proper plant spacing and removing infected leaves."
    }
}

# Language mapping for TTS
language_mapping = {
    'English': 'en',
    'Hindi': 'hi',
    'Telugu': 'te',
    'Tamil': 'ta',
    'Malayalam': 'ml',
    'Kannada': 'kn'
}

# Setting Title of App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

# Add Background Image CSS with Font Color Changes
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://static.toiimg.com/photo/resizemode-75,overlay-toiplus,msid-83832321/83832321.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: #ffffff; /* Default text color for the page */
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0); /* Transparent header */
    color: #ffffff; /* Header text color */
}
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.5); /* Optional sidebar transparency */
    color: #333333; /* Sidebar text color */
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff; /* Heading color */
}
p, div, span {
    color: #ffffff; /* General text color */
}

/* Style for the dropdown and file uploader */
.css-1wa3eu0-placeholder, .css-1k3dv2i {
    color: #ffffff !important; /* Text color for dropdown */
    background-color: rgba(0, 0, 0, 0.5) !important; /* Background with transparency */
}
.css-1q8dd3e.edgvbvh9 {
    color: #ffffff !important; /* Button text color */
    background-color: rgba(0, 0, 0, 0.5) !important; /* Button background with transparency */
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Language selection
selected_language = st.selectbox("Select Language:", list(language_mapping.keys()))

# Uploading the image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict Disease')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Displaying the image
        st.image(opencv_image, channels="BGR")
        # st.write(opencv_image.shape)
        
        # Resizing the image
        # opencv_image = cv2.resize(opencv_image, (256, 256))
        
        # Convert image to 4 Dimension
        opencv_image.shape = (1, 256, 256, 3)
        
        # Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
       
        # Handling plant and disease names
        plant_type = result.split('_')[0]  # Extract plant type
        disease = '_'.join(result.split('_')[1:])  # Join the remaining parts for the full disease name
        
        # Create translator object
        translator = Translator()

        # If plant is healthy
        if disease == "healthy":
            message = f"The {plant_type} leaf is healthy!"
            st.title(message)
            # Translate message
            try:
                translated_message = translator.translate(message, dest=language_mapping[selected_language]).text
                st.write("Translated message:", translated_message)
                speak_warning(translated_message, language_mapping[selected_language])
            except Exception as e:
                st.write("Translation error:", e)
        else:
            message = f"Warning! This is a {plant_type} leaf with {disease}."
            st.title(message)
            # Translate message
            try:
                translated_message = translator.translate(message, dest=language_mapping[selected_language]).text
                st.write("Translated message:", translated_message)
                speak_warning(translated_message, language_mapping[selected_language])
            except Exception as e:
                st.write("Translation error:", e)

            # Provide the URL for further information about the disease
            open_disease_info(plant_type, disease)

            # Display treatment and prevention information
            if result in treatment_info:  # Check using result (disease) instead of the variable 'disease'
                treatment_message = treatment_info[result]['treatment']
                prevention_message = treatment_info[result]['prevention']

                # Translate treatment and prevention messages
                try:
                    translated_treatment = translator.translate(treatment_message, dest=language_mapping[selected_language]).text
                    translated_prevention = translator.translate(prevention_message, dest=language_mapping[selected_language]).text

                    st.subheader("Treatment:")
                    st.write(translated_treatment)
                    speak_warning(translated_treatment, language_mapping[selected_language])

                    st.subheader("Prevention:")
                    st.write(translated_prevention)
                    speak_warning(translated_prevention, language_mapping[selected_language])
                except Exception as e:
                    st.write("Translation error:", e)
    else:
        st.error("No image uploaded. Please upload an image to proceed.")
