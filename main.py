import streamlit as st
import pickle
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()

# Download required NLTK data if not already present
nltk.download('stopwords')
nltk.download('punkt')


# Function to preprocess and transform the input text
def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Perform stemming
    text = [ps.stem(word) for word in text]

    # Join the words back into a single string
    return " ".join(text)


# Load the pre-trained models (ensure that the paths are correct)
try:
    tfidf = pickle.load(open('models/vectorizer.pkl', 'rb'))
    model = pickle.load(open('models/model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please check the file paths.")
    st.stop()

# Add logo image at the top corner with circular style using HTML/CSS
st.markdown(
    """
    <style>
    .logo-container {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        position: fixed;
        top: 0;
        left: 0;
        padding: 10px;
        z-index: 1;
    }
    .logo-container img {
        border-radius: 50%;
        width: 100px;
        height: 100px;
    }
    </style>
    <div class="logo-container">
        <img src="logo/sage.png" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)

# Streamlit application title
st.title("SMS Spam Classifier")

# Input text box for user to enter SMS
input_sms = st.text_input("Enter text to check spam or Not")

# Button to trigger prediction
if st.button('Predict'):
    if input_sms:
        # Preprocess the input SMS
        transformed_sms = transform_text(input_sms)

        # Vectorize the transformed SMS
        vect = tfidf.transform([transformed_sms])

        # Perform prediction using the loaded model
        result = model.predict(vect)[0]

        # Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.error("Please enter an SMS message to classify.")
