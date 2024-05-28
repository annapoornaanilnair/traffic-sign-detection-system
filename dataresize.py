import cv2
import os
import numpy as np

# Set the path to your dataset
dataset_path = "traffic_sign_classification_dataset\\train"

# Parameters
input_shape = (224, 224)


# Function to resize and preprocess images in the dataset
def preprocess_dataset(dataset_path, input_shape):
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Resize image
            image_resized = cv2.resize(image, input_shape)

            # Perform additional preprocessing steps, such as normalization
            image_normalized = image_resized / 255.0  # Normalize pixel values to [0, 1]

            # Save the resized and preprocessed image back to the dataset folder
            cv2.imwrite(
                image_path, (image_normalized * 255).astype(np.uint8)
            )  # Convert back to 0-255 range and save


# Preprocess the dataset
preprocess_dataset(dataset_path, input_shape)
