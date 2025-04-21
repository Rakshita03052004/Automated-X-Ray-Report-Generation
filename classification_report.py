import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import cv2

# Define paths
model_path = r"C:\Users\Lenovo\OneDrive\Desktop\dlproject\chest_xray\model\pneumonia_cnn.h5"
#test_images_dir = r"C:\Users\Lenovo\OneDrive\Desktop\dlproject\chest_xray\testcase_images\testcase2.jpeg"
# Path to the directory where test images are stored
test_images_dir = r"C:\Users\Lenovo\OneDrive\Desktop\dlproject\testcase_images"

# Load model
model = load_model(model_path)

# Function to load and preprocess images
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))  # Resize to the input size expected by your model
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Initialize lists to store the ground truth and predictions
y_true = []
y_pred = []

# Iterate through test images directory
for img_name in os.listdir(test_images_dir):
    img_path = os.path.join(test_images_dir, img_name)
    
    if img_name.endswith('.jpeg') or img_name.endswith('.jpg') or img_name.endswith('.png'):  # Only process image files
        # Get the ground truth label from the image name (Assumes your images are labeled like "pneumonia_1.jpeg")
        label = 1 if "pneumonia" in img_name.lower() else 0
        
        # Preprocess the image and make a prediction
        img = preprocess_image(img_path)
        pred = model.predict(img)
        pred_label = 1 if pred[0][0] > 0.5 else 0  # Assuming the model outputs a probability

        # Append true and predicted labels to the lists
        y_true.append(label)
        y_pred.append(pred_label)

# Generate the classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))
