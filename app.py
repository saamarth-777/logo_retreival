from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import Model
import pickle
from scipy.spatial.distance import euclidean

app = Flask(__name__)

# Load the pretrained VGG19 model
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extraction_model = Model(inputs=base_model.input, outputs=base_model.output)

# Load features dictionary
with open("features_dict.pkl", "rb") as f:
    features_dict = pickle.load(f)

# Route for the main page
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle the search functionality
@app.route("/search", methods=["POST"])
def search():
    if "image" not in request.files:
        return "No file uploaded!", 400

    # Load the uploaded image
    uploaded_file = request.files["image"]
    if uploaded_file.filename == "":
        return "No file selected!", 400

    # Save the uploaded image temporarily
    temp_path = "temp.jpg"
    uploaded_file.save(temp_path)

    # Preprocess the uploaded image
    img = tf.keras.preprocessing.image.load_img(temp_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))  # Add batch dimension

    # Extract features of the input image
    input_feature = feature_extraction_model.predict(img_array).flatten()

    # Compute similarity with stored features
    similarities = {}
    for label, features in features_dict.items():
        for idx, stored_feature in enumerate(features):
            dist = euclidean(input_feature, stored_feature)
            similarity = 100 * (1 - (dist / max(dist, 1e-6)))  # Avoid divide-by-zero errors
            similarities[f"{label} (image {idx + 1})"] = similarity

    # Sort by highest similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Remove the temporary image
    os.remove(temp_path)

    return render_template("results.html", results=sorted_similarities)

if __name__ == "__main__":
    app.run(debug=True)
