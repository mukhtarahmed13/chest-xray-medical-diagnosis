import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity
import os

random.seed(a=None, version=2)

set_verbosity(INFO)


def get_mean_std_per_batch(image_dir, df, H=320, W=320):
    sample_data = []
    for img in df.sample(100)["Image"].values:
        image_path = os.path.join(image_dir, img)
        sample_data.append(
            np.array(image.load_img(image_path, target_size=(H, W))))

    mean = np.mean(sample_data, axis=(0, 1, 2, 3))
    std = np.std(sample_data, axis=(0, 1, 2, 3), ddof=1)
    return mean, std


def load_image(img, image_dir, df, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    mean, std = get_mean_std_per_batch(image_dir, df, H=H, W=W)
    img_path = os.path.join(image_dir, img)
    x = image.load_img(img_path, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x


def grad_cam(input_model, image, cls, layer_name, H=320, W=320):
    """GradCAM method for visualizing input saliency."""
    import tensorflow as tf
    
    # Create a submodel to get both output and conv layer output
    conv_layer = input_model.get_layer(layer_name)
    submodel = tf.keras.models.Model([input_model.inputs], [input_model.output, conv_layer.output])
    
    with tf.GradientTape() as tape:
        model_out, conv_out = submodel(image)
        loss = model_out[:, cls]
    
    grads = tape.gradient(loss, conv_out)
    
    # Global average pooling of gradients
    weights = tf.reduce_mean(grads, axis=(1, 2))
    
    # Weighted sum of conv outputs
    cam = tf.reduce_sum(tf.multiply(conv_out, weights[:, tf.newaxis, tf.newaxis, :]), axis=-1)
    cam = cam[0]  # Remove batch dimension
    
    # Resize to original image size
    cam = cv2.resize(cam.numpy(), (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def compute_gradcam(model, img, image_dir, df, labels, selected_labels,
                    layer_name='bn'):
    preprocessed_input = load_image(img, image_dir, df)
    predictions = model.predict(preprocessed_input)

    print("Loading original image")
    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(load_image(img, image_dir, df, preprocess=False), cmap='gray')

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            print(f"Generating gradcam for class {labels[i]}")
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            plt.subplot(151 + j)
            plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
            plt.axis('off')
            plt.imshow(load_image(img, image_dir, df, preprocess=False),
                       cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
            j += 1


def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals


def load_data(train_csv='data/nih/train-small.csv', valid_csv='data/nih/valid-small.csv', test_csv='data/nih/test.csv', image_dir='data/nih/images-small/'):
    """Load train, validation and test datasets."""
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    test_df = pd.read_csv(test_csv)
    return train_df, valid_df, test_df


def preprocess_data(train_df, valid_df, test_df, image_dir, batch_size=32, target_size=(320, 320)):
    """Preprocess data using ImageDataGenerator."""
    from keras.preprocessing.image import ImageDataGenerator
    
    train_gen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    
    valid_gen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    
    test_gen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    
    labels = [c for c in train_df.columns if c not in ['Image', 'PatientId']]
    
    train_generator = train_gen.flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col='Image',
        y_col=labels,
        class_mode='raw',
        batch_size=batch_size,
        shuffle=True,
        target_size=target_size
    )
    
    valid_generator = valid_gen.flow_from_dataframe(
        dataframe=valid_df,
        directory=image_dir,
        x_col='Image',
        y_col=labels,
        class_mode='raw',
        batch_size=batch_size,
        shuffle=False,
        target_size=target_size
    )
    
    test_generator = test_gen.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col='Image',
        y_col=labels,
        class_mode='raw',
        batch_size=batch_size,
        shuffle=False,
        target_size=target_size
    )
    
    return train_generator, valid_generator, test_generator


def plot_history(history):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    
    plt.subplot(1, 2, 2)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_roc_curve(y_true, y_pred, labels):
    """Plot ROC curves for multiple classes."""
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def plot_pr_curve(y_true, y_pred, labels):
    """Plot Precision-Recall curves for multiple classes."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        avg_precision = average_precision_score(y_true[:, i], y_pred[:, i])
        plt.plot(recall, precision, label=f'{label} (AP = {avg_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
