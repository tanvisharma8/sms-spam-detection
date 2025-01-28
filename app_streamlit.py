# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:56:46 2025

@author: nishantsharma
"""

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import streamlit as st

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Data preprocessing functions
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

# Streamlit UI
st.title("SMS Spam Detection")
st.write("Enter an SMS message to classify it as Spam or Ham.")

# User input
user_input = st.text_area("Message", "")

if st.button("Predict"):
    if user_input:
        processed_data = preprocess_text(user_input)
        vectorized_data = vectorizer.transform([processed_data])
        prediction = model.predict(vectorized_data)
        
        if prediction[0] == 'spam':
            st.write("**Prediction:** Spam")
        else:
            st.write("**Prediction:** Ham")
    else:
        st.write("Please enter a message to classify.")
        