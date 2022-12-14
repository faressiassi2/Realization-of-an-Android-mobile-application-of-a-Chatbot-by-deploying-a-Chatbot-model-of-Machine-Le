
import import_ipynb
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np


with open("data.json", 'r') as f:
    datastore = json.load(f)

training_sentences = []
training_labels = []
labels = []
responses = []


for data in datastore['data']:
    for pattern in data['patterns']:
        training_sentences.append(pattern)
        training_labels.append(data['class'])
    responses.append(data['responses'])
    
    if data['class'] not in labels:
        labels.append(data['class'])

num_classes = len(labels)
num_classes


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train_labels = encoder.fit_transform(training_labels)
train_labels


#we define the hyperparametre:
vocab_size = 1000
embedding_dim = 16
max_length = 20
#oov_token = "<OOV>"


tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOF>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index


sequences = tokenizer.texts_to_sequences(training_sentences)

padded = pad_sequences(sequences,truncating='post',maxlen=max_length)

# you can define your own model architecture :
model = keras.Sequential([
    keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    #keras.layers.Flatten(),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16,activation='relu'),
    keras.layers.Dense(16,activation='relu'),
    keras.layers.Dense(num_classes,activation='softmax')
])


model.summary()


model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)


history = model.fit(
    padded,
    train_labels,
    epochs = 500
)


model5.save("chat4000_model")


import pickle
# to save the fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# to save the fitted label encoder
with open('encoder.pickle', 'wb') as ecn_file:
    pickle.dump(encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)


import colorama 
colorama.init()
from colorama import Fore, Style, Back
import json
import random
import pickle

with open("data.json") as file:
    datastore = json.load(file)


def chat():
    # load trained model
    model = keras.models.load_model('chat4000_model')

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break
            

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        
        tag = encoder.inverse_transform([np.argmax(result)])
    

        for i in datastore['data']:
            if i['class'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))
  

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()


filename = 'chat4000_model'
model = tf.saved_model.load(filename)

concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

concrete_func.inputs[0].set_shape([1,8])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.experimental_new_converter = True

Tflite_model = converter.convert()
open("chat4000.tflite", "wb").write(Tflite_model)


#converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
#tflite_model4 = converter.convert()
#open("chatbot4.tflite","wb").write(tflite_model4)



