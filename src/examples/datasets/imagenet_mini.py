from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import tensorflow as tf

def generate_dataset():
    #"""Generate the MNIST dataset with a validation split."""
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #x_train = x_train / 255.0
    #x_test = x_test / 255.0
    #y_train = to_categorical(y_train, 10)
    #y_test = to_categorical(y_test, 10)

    ## Split train data into train and validation sets
    #val_size = len(y_test)
    #x_val, y_val = x_train[:val_size], y_train[:val_size]
    #x_train, y_train = x_train[val_size:], y_train[val_size:]

    # Path to your ImageNet-Mini dataset
    dataset_path = './datasets/download/imagenet-mini'
    
    # Load training dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path + '/train',
        image_size=(224, 224),
        batch_size=128
    )
    
    # Get class names BEFORE prefetch
    class_names = train_dataset.class_names
    num_classes = len(class_names)
    
    # Load validation dataset
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path + '/val',
        image_size=(224, 224),
        batch_size=128
    )

    # Optional: Improve performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset, val_dataset

