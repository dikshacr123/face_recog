from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(data)
    pixel_string = data['pixels']
    pixels = np.array(list(map(float, pixel_string.split()))) \
                .reshape(48, 48, 1).astype('float32')

    pixels /= 255.0
    pixels = np.expand_dims(pixels, axis=0)  # shape: (1, 48, 48, 1)
    prediction = model.predict(pixels)
    emotion_idx = np.argmax(prediction)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    result = emotions[emotion_idx]
    return jsonify({'emotion': result})

if __name__ == '__main__':
    app.run(debug=True)
    