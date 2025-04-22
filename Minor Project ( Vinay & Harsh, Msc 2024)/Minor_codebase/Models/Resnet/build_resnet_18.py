from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_resnet18(input_shape, num_classes):
    """
    Builds a ResNet-18 model adapted for the given input shape and number of classes.
    """
    base_model = ResNet50(
        weights=None,  # Set to "imagenet" if pre-training is needed
        include_top=False,
        input_tensor=Input(shape=input_shape),
    )

    # Add custom classification layers on top
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        #loss='binary_crossentropy',
        metrics=["accuracy", "precision", "recall", "AUC"]
    )
    return model