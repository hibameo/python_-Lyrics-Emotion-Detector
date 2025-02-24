import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download VADER Lexicon
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Streamlit App Title
st.title("🎵 Lyrics Emotion Detector")

# User Input
lyrics = st.text_area("Paste song lyrics here:", "")

if st.button("Analyze Emotion"):
    if lyrics:
        # Sentiment Score Calculation
        sentiment_score = sia.polarity_scores(lyrics)

        # Emotion Detection Logic
        if sentiment_score["compound"] > 0.2:
            emotion = "😊 Happy"
        elif sentiment_score["compound"] < -0.2:
            emotion = "😢 Sad"
        else:
            emotion = "😐 Neutral"

        # Display Result
        st.subheader(f"Detected Emotion: {emotion}")

        # Visualization
        labels = ["Positive", "Neutral", "Negative"]
        scores = [sentiment_score["pos"], sentiment_score["neu"], sentiment_score["neg"]]

        fig, ax = plt.subplots()
        ax.bar(labels, scores, color=["green", "gray", "red"])
        ax.set_ylabel("Score")
        ax.set_title("Lyrics Sentiment Analysis")
        st.pyplot(fig)
