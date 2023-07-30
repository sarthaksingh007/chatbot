from flask import Flask, render_template, request, jsonify
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import random
#from playsound import playsound
import keras.models
#import tensorflow as tf
import numpy as np
import pandas as pd
#import json
#import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten
#from tensorflow.keras.models import Model
#import matplotlib.pyplot as plt
#import random
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
#from pydub import AudioSegment
#from pydub.playback import play
#import pygame



# Import your chatbot code here
#app = Flask(__name__, )

app = Flask(__name__,template_folder='template')

# Initialize the chatbot components
# ...

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.form['message']
    # Process the user's message using your chatbot code
    data=pd.read_csv('dataset_v-0-0-2.csv')
    le = LabelEncoder()
    y_train = le.fit_transform(data['tag'])

    input_shape=16
    vocabulary = 445
    output_length = 6
    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(data['input'])
    model=keras.models.load_model('modl3-0.h5')
    while True:
        texts_p = []
        prediction_input1 = message

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
            x=random.choice(data[data['tag']==response_tag.strip()]['responses'].reset_index(drop=True))+" "+str(covted)+" "+convert_to

            # # Convert text to speech
            # y=gtts.gTTS(x)
            # y.save("1.mp3")
            # print("Masiha : ",x)
            # song=AudioSegment.from_mp3("1.mp3")
            # play(song)
           #
        # Prepare the audio response
            #audio_response = AudioSegment.from_mp3('response.mp3')

        # Play the audio response
          #  play(audio_response)
            # sound = pygame.mixer.Sound('response.mp3')
            # sound.play()
            # pygame.time.wait(int(sound.get_length() * 1000))
            # pygame.mixer.quit()
            # pygame.quit()
            return jsonify({'response': x})
        elif floats:
            for value, index in floats:
                curr=value
                idx=index
                convert_from=prediction_input[idx+1]
                convert_to=prediction_input[idx+3]
            import requests

           #

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
            #print(response_tag)
            # print(data[data['tag']==response_tag.strip()]['input'])
            x=random.choice(data[data['tag']==response_tag.strip()]['responses'].reset_index(drop=True))+" "+str(covted)+" "+convert_to

            # # Convert text to speech
            # y=gtts.gTTS(x)
            # y.save("1.mp3")
            # print("Masiha : ",x)
            # song=AudioSegment.from_mp3("1.mp3")
            # play(song)
            #tts = gTTS(x)
            #tts.save('response.mp3')

        # Prepare the audio response
            #audio_response = AudioSegment.from_mp3('response.mp3')

        # Play the audio response
            #play(audio_response)
            # sound = pygame.mixer.Sound('response.mp3')
            # sound.play()
            # pygame.time.wait(int(sound.get_length() * 1000))
            # pygame.mixer.quit()
            # pygame.quit()
            return jsonify({'response': x})
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
                #print(response_tag)
            # print(data[data['tag']==response_tag.strip()]['input'])
                x=random.choice(data[data['tag']==response_tag.strip()]['responses'].reset_index(drop=True))
            #     #print(x)
            # # Convert text to speech
            #     y=gtts.gTTS(x)
            #     y.save("1.mp3")
            #    #playsound("1.mp3")
            #     print("Masiha : ",x)
            #     song=AudioSegment.from_mp3("1.mp3")
            #     play(song)
             # Convert text to speech
                #tts = gTTS(x)
                #tts.save('response.mp3')

        # Prepare the audio response
                #udio_response = AudioSegment.from_mp3('response.mp3')
                #play(audio_response)
                # sound = pygame.mixer.Sound('response.mp3')
                # sound.play()
                # pygame.time.wait(int(sound.get_length() * 1000))
                # pygame.mixer.quit()
                # pygame.quit()
                return jsonify({'response': x})
        # Play the audio response

                if response_tag=="goodbye":

                                break




if __name__ == '__main__':
    app.run()
