#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. Load Dataset ---

# Path to your ImageNet-Mini dataset
dataset_path = './datasets/download/imagenet-mini'

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path + '/train',
    image_size=(128, 128),
    batch_size=128
)

# Get class names BEFORE prefetch
class_names = train_dataset.class_names
num_classes = len(class_names)

# Load validation dataset
val_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path + '/val',
    image_size=(128, 128),
    batch_size=128
)

# Optional: Improve performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# --- 2. Define Model ---

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# --- 3. Compile Model ---

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 4. Train Model ---

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5
)

# --- 5. Evaluate ---

loss, acc = model.evaluate(val_dataset)
print(f"Validation Accuracy: {acc:.2f}")


