import os
import pickle
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import Model
from scipy.spatial.distance import euclidean

app = Flask(__name__)

# Load the feature dictionary
with open("features_dict.pkl", "rb") as f:
    features_dict = pickle.load(f)

# Load the VGG19 model
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extraction_model = Model(inputs=base_model.input, outputs=base_model.output)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    if "file" not in request.files:
        return "No file part in the request", 400
    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    # Save uploaded image temporarily
    temp_image_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(temp_image_path)

    # Preprocess the image
    img = load_img(temp_image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features for the uploaded image
    uploaded_image_features = feature_extraction_model.predict(img_array)
    uploaded_image_features = uploaded_image_features.flatten()

    # Search for similar images
    similarities = {}
    for label, feature_list in features_dict.items():
        for feature in feature_list:
            distance = euclidean(uploaded_image_features, feature)
            max_distance = np.sqrt(len(uploaded_image_features))  # Theoretical max distance
            similarity_percentage = (1 - (distance / max_distance)) * 100
            if label not in similarities:
                similarities[label] = []
            similarities[label].append(similarity_percentage)

    # Calculate average similarity for each label
    avg_similarities = {label: np.mean(values) for label, values in similarities.items()}

    # Sort by similarity percentage in descending order
    sorted_similarities = sorted(avg_similarities.items(), key=lambda x: x[1], reverse=True)

    # Cleanup temp image
    os.remove(temp_image_path)

    return render_template("results.html", similarities=sorted_similarities)

if __name__ == "__main__":
    app.run(debug=True)
