from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf

def cnn():
    # Define the model
    model = Sequential()

    # Add convolutional layers with appropriate arguments
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation="relu", padding="same",
                            input_shape=(224, 224, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation="relu", padding="same",
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation="relu", padding="same",
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())

    # Add the final output layer
    model.add(layers.Dense(units=512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(units=1, activation="sigmoid"))

    # Print a summary of the model
    model.summary()

    # Adjust the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model

