# src/main.py

# -------------------------------
# Step 1: Import Libraries
# -------------------------------
import os
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download NLTK stopwords (only first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


def main():
    print("Starting SMS spam detection pipeline...")

    # Prefer repository data path
    local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'spam_sms.csv'))
    alt_path = r"C:\Users\priya\Downloads\spam_sms.csv"

    if os.path.exists(local_path):
        dataset_path = local_path
    elif os.path.exists(alt_path):
        dataset_path = alt_path
    else:
        raise FileNotFoundError(
            f"Dataset not found. Checked: {local_path} and {alt_path}.\n" \
            "Please place spam_sms.csv in the data/ folder or update the path."
        )

    print(f"Loading dataset from: {dataset_path}")
    data = pd.read_csv(dataset_path)

    print("Dataset loaded. Sample:")
    print(data.head())
    print("\nLabel counts:")
    print(data['label'].value_counts())

    data.dropna(inplace=True)
    data['message_clean'] = data['message'].apply(preprocess)

    print("\nSample cleaned messages:")
    print(data[['message', 'message_clean']].head())

    data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        data['message_clean'], data['label_num'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    new_sms = [
        "Congratulations! You have won a free ticket!",
        "Hey, are we meeting today?",
        "Claim your free prize now!!!"
    ]
    new_sms_clean = [preprocess(text) for text in new_sms]
    new_sms_vec = vectorizer.transform(new_sms_clean)
    predictions = model.predict(new_sms_vec)

    for msg, pred in zip(new_sms, predictions):
        label = "Spam" if pred == 1 else "Ham"
        print(f"\nMessage: {msg}\nPrediction: {label}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("\nError: ", e)
        raise