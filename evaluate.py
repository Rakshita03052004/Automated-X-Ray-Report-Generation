import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("chest_xray/model/pneumonia_cnn.h5")

# Load test data
X_test = np.load("chest_xray/X_test.npy")
y_test = np.load("chest_xray/y_test.npy")

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")  # Convert probabilities to class labels

# Print Test Accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")




