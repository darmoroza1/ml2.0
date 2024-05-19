import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf


def train_nn(X_train, y_train, num_classes):
    y_train_enc = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

    model = tf.keras.models.Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_enc, epochs=10, batch_size=32, validation_split=0.2)

    return model


def evaluate_nn(model, X_test, y_test, num_classes):
    y_test_enc = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
    _, accuracy = model.evaluate(X_test, y_test_enc)
    return accuracy