from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import tensorflow as tf

def to_tf_dataset(x, y, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def generate_dataset(batch_size):
    """Generate the MNIST dataset with a validation split."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Split train data into train and validation sets
    val_size = len(y_test)
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]

    train_dataset = to_tf_dataset(x_train, y_train, batch_size)
    val_dataset = to_tf_dataset(x_val, y_val, batch_size)
    test_dataset = to_tf_dataset(x_test, y_test, batch_size)

    return train_dataset, val_dataset, test_dataset
