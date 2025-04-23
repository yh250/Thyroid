"""
Importing functions from all other files, cross-reference your directory structure and function names for any changes.
"""
from Models.Resnet.build_resnet_18 import build_resnet18
from Models.AlexNet.build_alexnet import build_alexnet
from Models.MobileNet.build_mobilenet_v2 import build_mobilenet
from Models.simple_cnn.build_simple_cnn import build_simple_cnn
from utils.save_results import create_results_folder, save_training_logs, save_loss_curve, save_model_checkpoint
from utils.Data_loader import *
# Importing Libraries here 
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
import os
import json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def main():
    """ Configuration
    Parameters: 
    input_shape = resolution of images and their colour channels 
    num_classes = number of classes 
    model_name = name of the model used in the current training cycle for logging purposes 
    augmentation_type = Name of the dataset/Augmentation type used in the current training cycle for logging purposes 
    """
    input_shape = (256, 256, 1)
    num_classes = 4
    #model_name = "Simple_CNN"
    model_name = "resnet18"

    augmentation_type = "simple"

    # Create results folder
    results_folder = create_results_folder(model_name=model_name, augmentation_type=augmentation_type)
    print(f"Results will be saved in: {results_folder}")

    #load data
    #train_generator, test_generator = load_data('/Users/harshyadav/Downloads/Minor_project/Dataset/Split_data/train','/Users/harshyadav/Downloads/Minor_project/Dataset/Split_data/val')
    #loading chest data to verify model accuracy
    #train_generator, test_generator = load_data(r"F:\Reena\Temp_Minor_project\Temp_Minor_project\SMOTE\train",
     #                                           r"F:\Reena\Temp_Minor_project\Temp_Minor_project\SMOTE\val")
    train_generator, test_generator = load_data(
        "/mnt/f/Reena/Temp_Minor_project/Temp_Minor_project/SMOTE/train",
        "/mnt/f/Reena/Temp_Minor_project/Temp_Minor_project/SMOTE/val"
    )

    #verify loaded data
    def print_generator_info(train_generator):
        """
        Prints the class labels and the number of images per class from the ImageDataGenerator.

        Parameters:
        - train_generator: The training data generator.
        """
        # Print class names and indices
        print("Class names and their corresponding indices:")
        for class_name, class_index in train_generator.class_indices.items():
            print(f"Class: {class_name}, Index: {class_index}")

        # Print the number of images in each class
        print("\nClass distribution (number of images in each class):")
        for class_name, num_images in train_generator.class_indices.items():
            class_folder = os.path.join(train_generator.directory, class_name)
            image_count = len([f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))])
            print(f"Class: {class_name}, Number of images: {image_count}")

    # Example usage
    print_generator_info(train_generator)

    """
    Callbacks for early stopping and best model saving
    Parameters: 
    monitor = which performance metric to keep track of  for early stopping of training 
    patience = Number of epochs for which we wait for any improvement in the monitor 
    NOTE: We only save model checkpoints in .keras format due to space constraints and storage efficiency. The 
    user may utilise their preferred way of storing the best model
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=5,  # Stop after N epochs of no improvement
        restore_best_weights=True  # Restore the best weights after stopping
    )

    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(results_folder, 'best_model.keras'),
        monitor='val_loss',
        save_best_only=True,  # Save only the best model
        mode='min',  # Save model with minimum validation loss
        verbose=1
    )

    """
    Build and train the model:
    Call the build function for the intended model architecture, for example, calling build_simple_cnn to use and train the 3-layer CNN. 
    Or call build_resnet18 to call the resnet model, which is stored in separate files that NEED TO BE RUN before calling the said model
    """
   #model = build_simple_cnn(input_shape, num_classes)
    model = build_resnet18(input_shape, num_classes)
    with tf.device('/GPU:0'):
        history = model.fit(
            train_generator,
            validation_data=(test_generator),
            epochs=10,
            batch_size=16,
            callbacks=[model_checkpoint, early_stopping]
        )

    # Store additional metrics in logs
    metrics_to_log = {
        "accuracy": history.history['accuracy'][-1],
        "val_accuracy": history.history['val_accuracy'][-1],
        "loss": history.history['loss'][-1],
        "val_loss": history.history['val_loss'][-1]
    }
    # Save results using functions in save_results
    save_training_logs(results_folder, metrics_to_log)
    save_loss_curve(results_folder, history)
    save_model_checkpoint(results_folder, model)
    
""" 
Plotting and saving the confusion matrix, ROC curve and Classfication Report
"""

    # Get predictions for the confusion matrix and ROC curve
    y_true = test_generator.classes
    y_pred_prob = model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_plot = plot_confusion_matrix(cm, class_names=train_generator.class_indices)
    cm_plot.savefig(os.path.join(results_folder, "confusion_matrix.png"))

    # Classification Report
""" 
Creates and saves a JSON file containing the classification report (Precision, Recall, etc), takes class labels from train_generator's indices
"""
    class_report = classification_report(y_true, y_pred, target_names=list(train_generator.class_indices.keys()),
                                         output_dict=True)
    with open(os.path.join(results_folder, "classification_report.json"), "w") as f:
        json.dump(class_report, f, indent=4)

    # ROC Curve
    fpr, tpr, roc_auc = plot_roc_curve(y_true, y_pred_prob, num_classes)
    plt.savefig(os.path.join(results_folder, "roc_curve.png"))

    print(f"Confusion matrix and ROC curve saved in: {results_folder}")


def plot_confusion_matrix(cm, class_names):
    """
    Plots and returns a confusion matrix heatmap using seaborn.
    """
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    plt.close()
    return plt


def plot_roc_curve(y_true, y_pred_prob, num_classes):
    """
    Plots and returns the ROC curve for multi-class classification.
    """
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(8, 8))

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    return fpr, tpr, roc_auc



if __name__ == "__main__":
    main()
