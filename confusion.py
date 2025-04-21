numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Correct model path
model = tf.keras.models.load_model("chest_xray/model/pneumonia_cnn.keras")

# Load test data
X_test = np.load("chest_xray/X_test.npy")
y_test = np.load("chest_xray/y_test.npy")

# Get predictions


# Get predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Use true labels directly if not one-hot encoded
y_true = y_test


# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


