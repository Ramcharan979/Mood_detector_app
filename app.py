import streamlit as st
from transformers import pipeline
from deepface import DeepFace
from PIL import Image
import os

# Text emotion detection pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Emoji mapping
emotion_emoji = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😲",
    "neutral": "😐",
    "disgust": "🤢",
    "love": "😍"
}

def detect_text_emotion(text):
    result = emotion_classifier(text)
    label = result[0]['label']
    return label

def detect_image_emotion(image_path):
    analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
    emotion = analysis[0]['dominant_emotion']
    return emotion

st.title("AI Mood Detector 🧠😊")
option = st.radio("Choose Input Type", ['Text', 'Image'])

if option == 'Text':
    user_text = st.text_input("Have fallen for u drlg")
    if user_text:
        emotion = detect_text_emotion(user_text.lower())
        st.write(f"Detected Emotion: **{emotion}** {emotion_emoji.get(emotion, '')}")

elif option == 'Image':
    uploaded_file = st.file_uploader("riyaz.jpg", type=['jpg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        emotion = detect_image_emotion("temp.jpg")
        st.write(f"Detected Emotion: **{emotion}** {emotion_emoji.get(emotion, '')}")
        os.remove("temp.jpg")
