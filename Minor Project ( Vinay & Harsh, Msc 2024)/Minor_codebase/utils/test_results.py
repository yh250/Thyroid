from utils.save_results import create_results_folder, save_training_logs, save_loss_curve, save_model_checkpoint
import tensorflow as tf
import numpy as np


# Define the main function
def main():
    # Create a results folder
    results_folder = create_results_folder()
    print(f"Results will be saved in: {results_folder}")

    # Example: Build and train a simple model
    input_shape = (128, 128, 1)
    num_classes = 4
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Dummy data
    train_data = np.random.rand(100, 128, 128, 1)
    train_labels = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 100), num_classes)

    # Train the model
    history = model.fit(train_data, train_labels, epochs=3, batch_size=32, validation_split=0.2)

    # Save results
    save_training_logs(results_folder, {"accuracy": history.history['accuracy'][-1],
                                        "val_accuracy": history.history['val_accuracy'][-1]})
    save_loss_curve(results_folder, history)
    save_model_checkpoint(results_folder, model)


# Call the main function
if __name__ == "__main__":
    main()
