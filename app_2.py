import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import Model

# Path to your directory with images
data_dir = "new_test_images"

# Ensure the directory has at least one subfolder, even if it doesn't matter
# This is required by image_dataset_from_directory
if not os.path.exists(os.path.join(data_dir, "placeholder")):
    os.makedirs(os.path.join(data_dir, "placeholder"))
    for file in os.listdir(data_dir):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            os.rename(os.path.join(data_dir, file), os.path.join(data_dir, "placeholder", file))

# Load images from the directory (no labels required)
dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels=None,  # No labels
    image_size=(224, 224),  # Resize to the size expected by VGG19
    batch_size=32,  # Adjust batch size based on your hardware
    shuffle=False  # Maintain order
)

# Preprocess the dataset for VGG19
def preprocess(image):
    return preprocess_input(image)

dataset = dataset.map(preprocess)

# Load the pretrained VGG19 model
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extraction_model = Model(inputs=base_model.input, outputs=base_model.output)

# Extract features for all images in the dataset
print("Extracting features...")
nfeatures = feature_extraction_model.predict(dataset, verbose=1)

# Print the shape of the extracted features
print(f"Shape of extracted features: {nfeatures.shape}")

# Flatten the features (if required) and save them as NumPy arrays
nflattened_features = nfeatures.reshape(nfeatures.shape[0], -1)  # Flatten to (num_images, features)
np.save("nflattened_features.npy", nflattened_features)

print(f"Features extracted and saved as 'nflattened_features.npy'. Shape: {nflattened_features.shape}")