import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Load your trained model
model = tf.keras.models.load_model("chest_xray/chest_xray/model/pneumonia_cnn.h5")

# 1. Print the model summary in the console
model.summary()

# 2. Save the model architecture as a PNG image
plot_model(model, to_file="cnn_model_architecture.png",show_shapes=True, show_layer_names=True)
