import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Load preprocessed data
X_train = np.load("chest_xray/X_train.npy")
y_train = np.load("chest_xray/y_train.npy")
X_val = np.load("chest_xray/X_val.npy")
y_val = np.load("chest_xray/y_val.npy")

# Normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_val = X_val / 255.0

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Define CNN model with Batch Normalization and L2 regularization
model = Sequential([
    Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(0.001), input_shape=(224, 224, 1)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

# Train the model with data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    validation_data=(X_val, y_val),
    epochs=50,
    callbacks=[early_stop, lr_reduce]
)

# Save the trained model
model.save("chest_xray/model/pneumonia_cnn.keras")
print("Model training complete. Saved to 'chest_xray/model/pneumonia_cnn.keras'")
