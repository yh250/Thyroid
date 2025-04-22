"""
Diagnostic tool for GPU availability in TensorFlow 
"""
import tensorflow as tf
print(tf.__version__)


# List available devices
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)


if physical_devices:

    # Restrict TensorFlow to use a specific GPU (if multiple GPUs are available)
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

    # Allow memory growth to prevent memory allocation issues
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found. Using CPU.")
