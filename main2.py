import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens and stop words
    filtered_tokens = [
        ps.stem(token) for token in tokens
        if token.isalnum() and token not in stopwords.words('english') and token not in string.punctuation
    ]

    return " ".join(filtered_tokens)

# Load the vectorizer and model with error handling
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms:
        with st.spinner("Processing..."):
            # 1. Preprocess the input text
            transformed_sms = transform_text(input_sms)
            # 2. Vectorize the preprocessed text
            vector_input = tfidf.transform([transformed_sms])
            # 3. Predict using the loaded model
            result = model.predict(vector_input)[0]
            # 4. Display the result
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
    else:
        st.warning("Please enter a message to classify.")