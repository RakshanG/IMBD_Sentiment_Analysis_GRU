import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

num_words = 30000
maxlen = 300

tokenizer = joblib.load("tokenizer.pkl")

model = load_model("imdb_gru_model.h5")

def encode_review(text):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=maxlen)

st.title("🎬 IMDB Movie Review Sentiment Analysis (GRU)")
st.write("Enter a movie review below and the model will predict whether it's Positive or Negative.")

user_input = st.text_area("📝 Your Review:")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("️Please enter a review.")
    else:
        encoded_input = encode_review(user_input)
        prediction = model.predict(encoded_input)[0][0]
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        st.subheader(f"Prediction: {sentiment}")
        st.write(f"Confidence Score: {prediction:.4f}")

