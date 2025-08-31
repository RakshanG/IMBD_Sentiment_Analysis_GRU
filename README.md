🎬 IMDB Movie Review Sentiment Analysis (GRU)

This project uses a **Gated Recurrent Unit (GRU)** deep learning model to classify IMDB movie reviews as **Positive 😀** or **Negative 😡**.  
It’s built with **TensorFlow/Keras** and deployed via **Streamlit** for interactive use.

---

🚀 Features
- GRU-based neural network for text classification  
- Tokenizer & padding for handling variable-length reviews  
- Streamlit web app for real-time predictions  
- Confidence scores for each prediction  

---

Project Structure
IMBD_Movie_Reviews/
│── app.py # Streamlit app for predictions
│── train_imdb_gru.py # Model training script (optional)
│── imdb_gru_model.h5 # Trained GRU model (ignored in GitHub if >100MB)
│── tokenizer.pkl # Saved tokenizer for preprocessing
│── imdb_word_index.json # Word index (if using local)
│── requirements.txt # Dependencies
│── README.md # Project documentation

---

Installation & Setup
Clone the repo:
```bash
git clone https://github.com/RakshanG/IMBD_Sentiment_Analysis_GRU.git
cd IMBD_Sentiment_Analysis_GRU

---

Run the App

Start the Streamlit app:
streamlit run app.py
Then open your browser at:
http://localhost:8501

---

Example

Input Review:
The movie was amazing, with brilliant acting and a fantastic storyline.

Prediction:
Positive 
Confidence Score: 0.95

---

Notes

The .h5 model file may be too large for GitHub. You can either:
Convert it to .tflite and include it in the repo, OR
Provide a Google Drive link for downloading the trained model.
To retrain, run train_imdb_gru.py.

---

Tech Stack

Python 3.11+
TensorFlow / Keras
Streamlit
Scikit-learn
NumPy / Pandas
