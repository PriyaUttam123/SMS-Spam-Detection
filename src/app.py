# src/app.py
import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Try to load trained model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), '..', 'spam_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'vectorizer.pkl')

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    # Train the model
    st.info("Training model... Please wait.")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spam_sms.csv')
    data = pd.read_csv(data_path)
    if 'v1' in data.columns:
        data = data.rename(columns={'v1': 'label', 'v2': 'message'})
    data.dropna(inplace=True)
    data['message_clean'] = data['message'].apply(preprocess)
    data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})
    X_train, X_test, y_train, y_test = train_test_split(
        data['message_clean'], data['label_num'], test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    st.success("Model trained and saved!")

# Streamlit UI
st.title("SMS Spam Classifier")
sms_input = st.text_area("Enter your SMS message:")

if st.button("Predict"):
    if sms_input.strip():
        sms_clean = preprocess(sms_input)
        sms_vec = vectorizer.transform([sms_clean])
        prediction = model.predict(sms_vec)[0]
        st.success("Spam" if prediction == 1 else "Ham")
    else:
        st.error("Please enter a message.")