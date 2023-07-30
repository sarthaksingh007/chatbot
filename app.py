import random

import keras.models
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import random
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
input_shape=16
vocabulary = 445
output_length = 6
model=keras.models.load_model('modl3-0.h5')
while True:
    texts_p = []
    prediction_input1 = input('You: ')
    
#removing punctuation and converting to lowercase    
    prediction_input = [letters.lower() for letters in prediction_input1 if letters not in string.punctuation]
    prediction_input =''.join(prediction_input)
    texts_p.append(prediction_input)
    prediction_input = prediction_input.strip().split(' ')
    integers = []
    floats = []

    for n, element in enumerate(prediction_input):
        try:
            integer_value = int(element)
            integers.append((integer_value, n))
        except ValueError:
            try:
                float_value = float(element)
                floats.append((float_value, n))
            except ValueError:
                pass

    if integers:
        for value, index in integers:
            curr=value
            idx=index
            convert_from=prediction_input[idx+1].upper()
            convert_to=prediction_input[idx+3].upper()
        import requests

        api_key = '97WwQWGRmUjka1Qk2BZjZyGKRRN2WCE5'
        endpoint = f'https://api.apilayer.com/currency_data/convert?to={convert_to}&from={convert_from}&amount={curr}'

        headers = {
            'apikey': api_key
            }

        respon = requests.get(endpoint, headers=headers)
        s = respon.json()
        covted=s['result']
        
        prediction_input = [letters.lower() for letters in prediction_input1 if letters not in string.punctuation]
        prediction_input =''.join(prediction_input)
        texts_p.append(prediction_input)

        #tokenizing and padding
        prediction_input = tokenizer.texts_to_sequences(texts_p)
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input], input_shape)
        #getting output from model


        output = model.predict(prediction_input)
        output = output.argmax()
        #finding the right tag and predicting
        response_tag = le.inverse_transform([output])[0]
        print(response_tag)
        # print(data[data['tag']==response_tag.strip()]['input'])
        print("Masiha : ",random.choice(data[data['tag']==response_tag.strip()]['responses'].reset_index(drop=True)),covted,convert_to)
    elif floats:
        for value, index in floats:
            curr=value
            idx=index
            convert_from=prediction_input[idx+1]
            convert_to=prediction_input[idx+3]
        import requests

        api_key = '97WwQWGRmUjka1Qk2BZjZyGKRRN2WCE5'
        endpoint = f'https://api.apilayer.com/currency_data/convert?to={convert_to}&from={convert_from}&amount={curr}'

        headers = {
            'apikey': api_key
            }

        respon = requests.get(endpoint, headers=headers)
        s = respon.json()
        covted=s['result']
        prediction_input = [letters.lower() for letters in prediction_input1 if letters not in string.punctuation]
        prediction_input =''.join(prediction_input)
        texts_p.append(prediction_input)

        #tokenizing and padding
        prediction_input = tokenizer.texts_to_sequences(texts_p)
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input], input_shape)
        #getting output from model


        output = model.predict(prediction_input)
        output = output.argmax()
        #finding the right tag and predicting
        response_tag = le.inverse_transform([output])[0]
        print(response_tag)
        # print(data[data['tag']==response_tag.strip()]['input'])
        print("Bot : ",random.choice(data[data['tag']==response_tag.strip()]['responses'].reset_index(drop=True)),covted,convert_to)
    else:
            prediction_input = [letters.lower() for letters in prediction_input1 if letters not in string.punctuation]
            prediction_input =''.join(prediction_input)
            texts_p.append(prediction_input)

        #tokenizing and padding
            prediction_input = tokenizer.texts_to_sequences(texts_p)
            prediction_input = np.array(prediction_input).reshape(-1)
            prediction_input = pad_sequences([prediction_input], input_shape)
        #getting output from model


            output = model.predict(prediction_input)
            output = output.argmax()
        #finding the right tag and predicting
            response_tag = le.inverse_transform([output])[0]
            print(response_tag)
        # print(data[data['tag']==response_tag.strip()]['input'])
            print("Bot : ",random.choice(data[data['tag']==response_tag.strip()]['responses'].reset_index(drop=True)))
            if response_tag=="goodbye":
                break   
