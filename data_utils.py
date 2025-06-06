# data_utils.py
import tensorflow as tf

IMG_HEIGHT = 128
IMG_WIDTH  = 128
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE

def get_datasets(image_dir="images"):
    """
    Loads images from `image_dir`, splits 80/20, normalizes, augments train, caches/prefetches.
    Returns: (train_ds, val_ds), both tf.data.Dataset objects.
    """
    # 1) Load & split
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 2) Normalize pixel values
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y),   num_parallel_calls=AUTOTUNE)

    # 3) Data augmentation on training only
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                             num_parallel_calls=AUTOTUNE)

    # 4) Cache & prefetch
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds
