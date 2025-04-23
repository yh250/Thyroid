import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(train_dir, test_dir, batch_size=32, target_size=(256, 256)):
    """
    Loads the dataset using ImageDataGenerator and returns train and test generators.

    Parameters:
    - train_dir: str, path to the training dataset directory.
    - test_dir: str, path to the testing dataset directory.
    - batch_size: int, number of samples per batch.
    - target_size: tuple, the target size to which images will be resized.

    Returns:
    - train_generator: DataGenerator, generator for training data.
    - test_generator: DataGenerator, generator for testing data.
    """
    # Initialize ImageDataGenerators
    train_datagen = ImageDataGenerator(

        rescale=1. / 255,  # Normalize pixel values to [0, 1]
        rotation_range=40,  # Randomly rotate images
        width_shift_range=0.2,  # Randomly shift images horizontally
        height_shift_range=0.2,  # Randomly shift images vertically
        shear_range=0.2,  # Randomly apply shear transformations
        zoom_range=0.2,  # Randomly zoom images
        horizontal_flip=True,  # Randomly flip images horizontally
        fill_mode='nearest'  # Fill empty pixels after transformations
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)  # Only normalization for test data

    # Create generators for training and testing data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',  # For multi-class classification
        shuffle=True,
        color_mode = 'grayscale'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,  # No need to shuffle test data
        color_mode='grayscale'
    )

    return train_generator, test_generator


# Example usage
#train_generator, test_generator = load_data(
    #train_dir="path/to/output/train",
    #test_dir="path/to/output/test"
#)
