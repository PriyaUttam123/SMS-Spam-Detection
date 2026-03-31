# src/app.py
import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    st.success("Model trained and saved!")

# Streamlit UI
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📩 SMS Spam Detector")
st.markdown(
    """
    **Predict whether a message is Spam or Ham** using a TF-IDF + Logistic Regression model.
    Enter your SMS text below and click **Predict**.

    - `Ham`: normal message
    - `Spam`: unsolicited/phishing/fraud message
    """
)

with st.sidebar:
    st.header("Settings")
    max_features = st.slider("TF-IDF max features", min_value=1000, max_value=10000, value=5000, step=500)
    re_train = st.button("Retrain model")
    st.markdown("---")
    st.markdown("### Example messages")
    st.write("- Free entry in 2 a wkly comp to win FA Cup final tickets")
    st.write("- Hey, are we still meeting for lunch?")
    st.write("- You have won a cash prize! Call now.")

# Reload model + vectorizer if settings changed
if re_train:
    st.info("Retraining model with new settings...")
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
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    st.success("Retrained and saved model!")

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.warning("Model files not found, training now...")
else:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

sms_input = st.text_area("Enter your SMS message:", height=140)

if st.button("Predict"):
    if sms_input.strip():
        sms_clean = preprocess(sms_input)
        st.text_area("Cleaned message", value=sms_clean, height=120)
        sms_vec = vectorizer.transform([sms_clean])
        prediction = model.predict(sms_vec)[0]
        prob = model.predict_proba(sms_vec)[0]

        result = "Spam" if prediction == 1 else "Ham"
        color = "red" if result == "Spam" else "green"
        st.markdown(f"<h2 style='color: {color};'>Prediction: {result}</h2>", unsafe_allow_html=True)
        st.write(f"Spam probability: {prob[1]:.3f}")
        st.write(f"Ham probability: {prob[0]:.3f}")

        if prob[1] > 0.75:
            st.warning("High confidence spam prediction — be careful!")
        elif prob[0] > 0.75:
            st.success("High confidence ham prediction.")
        else:
            st.info("Low confidence. Consider retraining with more data.")
    else:
        st.error("Please enter a message to evaluate.")
