def build_mobilenet(input_shape, num_classes):
    """
    Builds a MobileNet model for small datasets.
    """
    base_model = MobileNet(input_shape=input_shape, include_top=False, weights=None)
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
