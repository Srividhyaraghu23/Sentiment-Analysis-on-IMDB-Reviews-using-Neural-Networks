import streamlit as st
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model("movie_sentiment_model.h5")

# Load tokenizer
tokenizer = joblib.load("tokenizer.pkl")

st.title("🎬 IMDB Movie Sentiment Analysis")
st.write("Enter a movie review to predict whether it is Positive or Negative.")

review = st.text_area("Movie Review")

if st.button("Predict Sentiment"):

    # Convert review to sequence
    sequence = tokenizer.texts_to_sequences([review])

    # Pad sequence
    padded = pad_sequences(
        sequence,
        maxlen=200,
        padding="post",
        truncating="post"
    )

    # Predict
    prediction = model.predict(padded)

    score = prediction[0][0]

    if score >= 0.5:
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative 😞"

    st.success(f"Predicted Sentiment: {sentiment}")
    st.write(f"Confidence Score: {score:.4f}")