import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Define the transform_text function
def transform_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Tokenize the text
    text = text.split()
    
    # Remove stopwords and apply stemming
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    
    # Join the words back into one string
    text = ' '.join(text)
    
    return text

# Load the saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app code starts here
st.title("Email Spam Classifier")
input_sms = st.text_area("Enter message")

if st.button('Predict'):
    # Preprocess the input message
    transformed_sms = transform_text(input_sms)
    # Vectorize the transformed message
    vector_input = tfidf.transform([transformed_sms])
    # Predict whether it is spam or not
    result = model.predict(vector_input)[0]
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
