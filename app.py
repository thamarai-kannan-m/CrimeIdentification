import numpy as np
import os
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Loading the model
model = load_model("Crime_classification.h5")

app = Flask(__name__)

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

# About page route
@app.route('/about')
def about():
    return render_template('about.html')

# Predict page route
@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result', methods=["GET", "POST"])
def res():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)  # Getting the current path where app.py is present
        filepath = os.path.join(basepath, 'uploads', f.filename)  # Saving the uploaded image to the 'uploads' folder
        f.save(filepath)

        img = image.load_img(filepath, target_size=(64, 64))  # Reading the image
        x = image.img_to_array(img)  # Converting the image into an array
        x = np.expand_dims(x, axis=0)  # Expanding Dimensions
        pred = np.argmax(model.predict(x))  # Predicting the higher probability index

        op = ['Fighting', 'Arrest', 'Vandalism', 'Assault', 'Stealing', 'Arson', 'NormalVideos', 'Burglary', 'Explosion',
              'Robbery', 'Abuse', 'Shooting', 'Shoplifting', 'RoadAccidents']  # Creating a list
        result = op[pred]

        return render_template('prediction.html', pred=result)

if __name__ == '__main__':
    app.run()


