import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
from tensorflow.keras.preprocessing.image import DirectoryIterator


class IntervalLogger(tf.keras.callbacks.ProgbarLogger):
    """
    Custom callback for logging training progress at specified intervals.
    This helps reduce console output by only showing progress at certain epoch intervals.
    """
    def __init__(self, verbose: int = 1, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.default_verbose = verbose
        self.epoch_interval = None

    def on_train_begin(self, logs: Optional[Dict[str, float]] = None) -> None:
        super().on_train_begin(logs)
        total_epochs = self.epochs
        if total_epochs is None:
            self.epoch_interval = 10
        else:
            self.epoch_interval = max(1, total_epochs // 10)

    def on_epoch_begin(self, epoch: int, *args: Any, **kwargs: Any) -> None:
        if epoch == 0 or (epoch + 1) % self.epoch_interval == 0:
            self.verbose = self.default_verbose
        else:
            self.verbose = 0
        super().on_epoch_begin(epoch, *args, **kwargs)


def train_dataset(batch_size: int = 64) -> Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]:
    """
    Prepare and load the training, validation, and test datasets with data augmentation.
    
    Args:
        batch_size (int): Number of samples per batch
        
    Returns:
        tuple: (train_generator, validation_generator, test_generator)
    """
    dir_train = '../PetImages/train'
    dir_test = '../PetImages/val'

    # Data augmentation configuration for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.3,
        rotation_range=50,
        width_shift_range=0.1,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.1,
    )
    # Only rescale test data, no augmentation
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load training data with augmentation
    train = train_datagen.flow_from_directory(
        dir_train,
        target_size=(180, 180),
        batch_size=batch_size,
        subset="training",
        class_mode='categorical'
    )
    # Load validation data
    val = train_datagen.flow_from_directory(
        dir_train,
        target_size=(180, 180),
        batch_size=batch_size,
        subset="validation",
        class_mode='categorical'
    )
    # Load test data
    test = test_datagen.flow_from_directory(
        dir_test,
        target_size=(180, 180),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train, val, test


def CNN() -> Sequential:
    """
    Create and return a CNN model for pet image classification.
    
    Returns:
        Sequential: A CNN model with the following architecture:
        - Three convolutional blocks with increasing filters
        - Batch normalization and max pooling after each block
        - Dense layers with dropout for classification
    """
    model = Sequential()

    # Input Layer
    model.add(Input(shape=(180, 180, 3)))
    
    # First Convolutional Block with 128 filters
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    # Second Convolutional Block with 256 filters
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    # Third Convolutional Block with 512 filters
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation="softmax"))
    
    return model


def train_dataset(batch_size: int = 32) -> Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]:
    dir_train = '../PetImages/train'
    dir_test = '../PetImages/val'

    # Пом'якшена аугментація (прибрано vertical_flip, зменшено повороти та зум)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.15,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.1,
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train = train_datagen.flow_from_directory(
        dir_train,
        target_size=(180, 180),
        batch_size=batch_size,
        subset="training",
        class_mode='categorical'
    )
    val = train_datagen.flow_from_directory(
        dir_train,
        target_size=(180, 180),
        batch_size=batch_size,
        subset="validation",
        class_mode='categorical'
    )
    test = test_datagen.flow_from_directory(
        dir_test,
        target_size=(180, 180),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train, val, test


def train(batch_size: int = 32, epochs: int = 60) -> Tuple[Sequential, DirectoryIterator]:
    train_data, val_data, test_data = train_dataset(batch_size=batch_size)
    
    model = CNN()
    # Зменшуємо learning_rate з 0.001 до 0.0003 для стабільного виходу з плато
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Зберігаємо найкращу модель за val_accuracy та плавно зменшуємо lr на плато
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_pet_cnn.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[IntervalLogger(), checkpoint, reduce_lr],
        verbose=0
    )

    # Завантажуємо найкращі ваги перед фінальною оцінкою
    model = tf.keras.models.load_model("best_pet_cnn.keras")
    
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test accuracy: {test_acc:.4f}")
    
    return model, test_data


def visualize_augmented_images(train_data: DirectoryIterator) -> None:
    """
    Visualize a batch of augmented training images with their labels.
    
    Args:
        train_data: Training data generator
    """
    images, labels = next(train_data)
    class_names = {0: 'Cat', 1: 'Dog'}
    
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[np.argmax(labels[i])])
    plt.tight_layout()


def visualize_predictions(model: Sequential, test_data: DirectoryIterator) -> None:
    """
    Visualize model predictions on test images.
    
    Args:
        model: Trained CNN model
        test_data: Test data generator
    """
    images, true_labels = next(test_data)
    predictions = model.predict(images)
    class_names = {0: 'Cat', 1: 'Dog'}
   
    plt.figure(figsize=(14, 16))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        
        predicted_class = class_names[np.argmax(predictions[i])]
        true_class = class_names[np.argmax(true_labels[i])]

        color = "green" if predicted_class == true_class else "red"
        plt.xlabel(f"Pred: {predicted_class}\nTrue: {true_class}", color=color, fontsize=11)
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()

if __name__ == '__main__':
    # Load data with augmentation
    train_data, val_data, test_data = train_dataset()
    
    # Visualize augmented images
    visualize_augmented_images(train_data)
    
    # Train the model
    model, test_data = train()
    
    # Visualize predictions
    visualize_predictions(model, test_data)