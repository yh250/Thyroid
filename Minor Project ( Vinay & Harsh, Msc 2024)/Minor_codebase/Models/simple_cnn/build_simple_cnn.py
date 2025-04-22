# models/simple_cnn.py
import tensorflow as tf

def build_simple_cnn(input_shape, num_classes):
    """
    Builds a CNN with three convolutional layers, max-pooling, dropout,
    and two dense layers for classification.

    Parameters:
    - input_shape: tuple, shape of the input data (height, width, channels).
    - num_classes: int, number of output classes.

    Returns:
    - model: tf.keras.Model, the compiled CNN model.
    """
    model = tf.keras.Sequential([
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten the feature maps
        tf.keras.layers.Flatten(),

        # Fully Connected Layers
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Dropout layer to reduce overfitting
        tf.keras.layers.Dense(64, activation='relu'),

        # Output Layer
        tf.keras.layers.Dense(num_classes, activation='softmax')  # For multi-class classification
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
