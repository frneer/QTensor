import kagglehub

from pathlib import Path
import tensorflow as tf



def generate_dataset(batch_size):
    # Download latest version
    output_path = Path(__file__).parent.parent / "downloads"
    dataset_path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")

    # Load training dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path + '/imagenet-mini/train',
        image_size=(128, 128),
        batch_size=batch_size
    )

    # Load validation dataset
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path + '/imagenet-mini/val',
        image_size=(128, 128),
        batch_size=batch_size
    )

    # Create test dataset from train dataset
    train_dataset = train_dataset.skip(len(val_dataset)).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = train_dataset.take(len(val_dataset)).prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_dataset, val_dataset, test_dataset
