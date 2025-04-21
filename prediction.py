import os
import cv2
import numpy as np
#from tensorflow.keras.models import load_model

# Path to the image
img_path = r"C:\Users\Lenovo\OneDrive\Desktop\dlproject\chest_xray\testcase_images\testcase2.jpeg"

# Check if image exists
if not os.path.exists(img_path):
    print(f"❌ Image file not found at: {img_path}")
    print("✅ Available files in the directory:")
    print(os.listdir(os.path.dirname(img_path)))
    exit()

# Load model
model_path = r"C:\Users\Lenovo\OneDrive\Desktop\dlproject\chest_xray\model\pneumonia_cnn.h5"
#model = load_model(model_path)

# Preprocess the image
img = cv2.imread(img_path)

# Convert the image to grayscale (1 channel)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize the image to match model input size (222x222)
img = cv2.resize(img, (222, 222))

# Normalize the pixel values
img = img / 255.0

# Add the channel dimension (grayscale image has 1 channel)
img = np.expand_dims(img, axis=-1)  # Shape becomes (222, 222, 1)

# Add the batch dimension (for prediction, we need shape (batch_size, height, width, channels))
img = np.expand_dims(img, axis=0)   # Shape becomes (1, 222, 222, 1)

# Check the image shape before prediction
print(f"Image shape after preprocessing: {img.shape}")

# Make prediction
#pred = model.predict(img)
#label = "Pneumonia" if pred[0][0] > 0.5 else "Normal"
#print(f"✅ Prediction: {label}")
