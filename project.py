import os
import cv2
import matplotlib.pyplot as plt

# Define dataset paths
data_dir = "C:/Users/Lenovo/OneDrive/Desktop/dlproject/chest_xray/"
train_dir = os.path.join(data_dir, "train")

# Get sample images
categories = ["NORMAL", "PNEUMONIA"]
for category in categories:
    folder = os.path.join(train_dir, category)
    first_image = os.listdir(folder)[0]  # Get first image
    image_path = os.path.join(folder, first_image)

    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Display image
    plt.figure()
    plt.imshow(img)
    plt.title(category)
    plt.axis("off")

plt.show()
