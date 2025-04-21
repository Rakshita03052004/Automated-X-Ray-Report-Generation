import os
import numpy as np
import tensorflow as tf
#from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

#from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Define image dimensions
IMG_SIZE = (224, 224)

# Define paths
train_dir = "chest_xray/train"
test_dir = "chest_xray/test"
val_dir = "chest_xray/val"


# Function to load images & labels
def load_data(directory):
    images, labels = [], []
    for label, condition in enumerate(["NORMAL", "PNEUMONIA"]):  # 0 = Normal, 1 = Pneumonia
        path = os.path.join(directory, condition)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
            img_array = img_to_array(img) / 255.0  # Normalize
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load train, validation, and test data
X_train, y_train = load_data(train_dir)
X_val, y_val = load_data(val_dir)
X_test, y_test = load_data(test_dir)

# Save preprocessed data
np.save("chest_xray/X_train.npy", X_train)
np.save("chest_xray/y_train.npy", y_train)
np.save("chest_xray/X_val.npy", X_val)
np.save("chest_xray/y_val.npy", y_val)
np.save("chest_xray/X_test.npy", X_test)
np.save("chest_xray/y_test.npy", y_test)

print("Preprocessing complete. Data saved!")


