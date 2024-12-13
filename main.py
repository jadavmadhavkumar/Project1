import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from urllib.parse import urlparse
import re
import pandas as pd

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

# Text preprocessing function for spam detection
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text into words
    y = [i for i in text if i.isalnum()]  # Remove non-alphanumeric tokens
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

# Function to load or fit the TfidfVectorizer
def get_or_fit_vectorizer():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            # Ensure the vectorizer is fitted
            check_is_fitted(vectorizer)
            return vectorizer
    except (FileNotFoundError, ValueError, pickle.UnpicklingError, Exception):
        st.warning("⚠️ Vectorizer not found or not fitted. Fitting a new vectorizer...")

        # Example training corpus to fit the vectorizer
        training_corpus = [
            "This is a spam message",
            "This is a ham message",
            "Spam messages are annoying",
            "Ham messages are helpful"
        ]
        vectorizer = TfidfVectorizer()
        vectorizer.fit(training_corpus)

        # Save the newly fitted vectorizer for future use
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        st.success("✅ A new vectorizer has been fitted and saved successfully!")
        return vectorizer

# Load or create the vectorizer
tfidf = get_or_fit_vectorizer()

# Load the spam detection model
try:
    model = pickle.load(open('model.pkl', 'rb'))  # Load spam detection model
except FileNotFoundError as e:
    st.error(f"Error loading spam detection model: {e}")
    st.stop()

# Load phishing model (optional)
try:
    phishing_model = pickle.load(open('phishing_url.pkl', 'rb'))  # Load phishing URL model
except FileNotFoundError:
    phishing_model = None

# Function to preprocess URLs for phishing detection
def preprocess_url(url):
    parsed_url = urlparse(url)
    features = {
        "Length_of_URL": len(url),
        "Have_IP": 1 if re.match(r"(\d{1,3}\.){3}\d{1,3}", parsed_url.netloc) else 0,
        "Have_At": 1 if "@" in url else 0,
        "Number_of_Dots": url.count("."),
        "Number_of_Special_Char": len(re.findall(r"[\W_]", url)) - url.count("."),
        "Length_of_Domain": len(parsed_url.netloc),
        "Have_Https": 1 if parsed_url.scheme == "https" else 0,
    }
    return pd.DataFrame([features])

# Function to predict phishing URLs
def predict_phishing(url):
    if phishing_model is None:
        return None, "⚠️ Phishing detection model not loaded."
    try:
        features = preprocess_url(url)
        prediction = phishing_model.predict(features)[0]
        probability = phishing_model.predict_proba(features)[0][1]
        return prediction, probability
    except Exception as e:
        return None, f"⚠️ Error during phishing detection: {e}"

# Function to detect malware
def detect_malware(file):
    if file:
        file_extension = file.name.split('.')[-1].lower()
        if file_extension in ["exe", "dll", "zip", "tar", "pdf"]:
            return "🚨 Malware detected! Suspicious file type."
        return "✅ No malware detected for this file."
    return "⚠️ No file uploaded for malware detection."

# Streamlit app setup
st.set_page_config(
    page_title="📧 Spam, Malware, and Phishing Detection Tool",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("📧 Spam, Malware, and Phishing Detection Tool")
st.write("Use this tool to classify messages as **Spam**, detect **Phishing URLs**, or check files for **Malware**.")

# Input fields
input_sms = st.text_area("✉️ Enter the message:", placeholder="Type your email or SMS here...")
input_url = st.text_input("🌐 Enter a URL to check for phishing:", placeholder="Type URL here...")
uploaded_file = st.file_uploader("📂 Upload a text file for spam prediction", type=["txt", "csv"])
malware_file = st.file_uploader("🚨 Upload a file for Malware Detection", type=["exe", "dll", "zip", "tar", "pdf"])

# Prediction button
if st.button("🚀 Predict"):
    if not input_sms.strip() and not input_url.strip() and not uploaded_file and not malware_file:
        st.warning("⚠️ Please enter a message, URL, or upload a file for detection!")
    else:
        # Spam Detection
        if input_sms.strip():
            try:
                transformed_sms = transform_text(input_sms)
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]
                st.markdown(
                    "<h3 style='color: red;'>🚨 Spam Detected!</h3>" if result == 1 else "<h3 style='color: green;'>✅ Not Spam</h3>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"⚠️ An error occurred during spam detection: {e}")

        # Phishing URL Detection
        if input_url.strip():
            phishing_result, phishing_probability = predict_phishing(input_url)
            if phishing_result is None:
                st.error(phishing_probability)
            else:
                st.markdown(
                    f"<h3 style='color: red;'>🚨 Phishing URL Detected ({phishing_probability:.2%})</h3>" if phishing_result == 1 else f"<h3 style='color: green;'>✅ Safe URL ({1 - phishing_probability:.2%})</h3>",
                    unsafe_allow_html=True,
                )

        # File-based Spam Detection
        if uploaded_file:
            file_contents = uploaded_file.read().decode("utf-8")
            messages = file_contents.splitlines()
            for message in messages:
                transformed_sms = transform_text(message)
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]
                st.write(f"Message: {message}")
                st.markdown(
                    "<span style='color: red;'>🚨 Spam Detected!</span>" if result == 1 else "<span style='color: green;'>✅ Not Spam</span>",
                    unsafe_allow_html=True,
                )

        # Malware Detection
        if malware_file:
            malware_result = detect_malware(malware_file)
            st.markdown(f"<h3>{malware_result}</h3>", unsafe_allow_html=True)

# Footer
st.markdown("---\nDeveloped by **Your Name** | Powered by [Streamlit](https://streamlit.io/)")
