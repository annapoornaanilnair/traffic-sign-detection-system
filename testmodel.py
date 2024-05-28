from cvzone.ClassificationModule import Classifier
import cv2
import os

# Load the Classifier object
Classifier = Classifier("mymodel/keras_model.h5", "mymodel/labels.txt")

# Define the path to the dataset
dataset_path = "traffic_sign_classification_dataset\\train"

# Loop through the images in the dataset directory
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg") or filename.endswith(
        ".png"
    ):  # Assuming images are JPEG or PNG
        # Read the image
        img = cv2.imread(os.path.join(dataset_path, filename))

        # Get prediction for the image
        prediction, index = Classifier.getPrediction(img)

        # Print the prediction result in the terminal
        print(f"Image: {filename}, Prediction: {prediction}")
