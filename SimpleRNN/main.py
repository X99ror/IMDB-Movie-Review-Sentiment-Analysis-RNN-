import numpy as np 
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import streamlit as st

word_index=imdb.get_word_index()
reverse_word_index={value:key for key, value in word_index.items()}

model= load_model('SimpleRNN/simple_rnn_imdb.h5')


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(text,max_features=10000):
    words = text.lower().split()
    encoded_review = []
    for word in words:
        index = word_index.get(word, 2) + 3  
        if index < max_features:
            encoded_review.append(index)
        else:
            encoded_review.append(2)
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


def predict_sentiment(review):
    processed_input=preprocess_text(review)
    prediction=model.predict(processed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment , prediction[0][0]

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')


use_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed=preprocess_text(use_input)
    
    prediction=model.predict(preprocessed)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    st.write(f'Sentiment:{sentiment}')
    st.write(f'Prediction Score:{prediction[0][0]}')
else:
    st.write('Please enter a movie review')