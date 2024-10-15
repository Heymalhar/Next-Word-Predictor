import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

model = tf.keras.models.load_model('next_word_predictor_model.h5')

with open('data.txt', 'r', encoding='utf-8') as file:
    text = file.read()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

def predict_next_words(model, tokenizer, seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        seed_text += " " + output_word
    
    return seed_text

st.title('Next Word Prediction App')

with st.expander("Introduction to the App"):
    st.write("""
        This app predicts the next words in a given sentence using an LSTM model.
        You can input a seed text and select the number of words you'd like the model to predict.
        Upon clicking the 'Predict' button, the app will generate the next words based on the input.
        The model has been trained on a text corpus (A Sherlock Holmes Book) to understand language patterns and predict the next possible words.
    """)

seed_text = st.text_input("Enter the seed text:")
next_words = st.number_input("Enter the number of words to predict:", min_value=1, max_value=100, value=3)

if st.button('Predict'):
    if seed_text:
        max_sequence_len = max([len(tokenizer.texts_to_sequences([line])[0]) for line in text.split('\n')])
        generated_text = predict_next_words(model, tokenizer, seed_text, next_words, max_sequence_len)
        st.write("Generated text:", generated_text)
    else:
        st.write("Please enter seed text to predict.")
