#!/usr/bin/env python3
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Conv2D, ELU, Dropout
from keras.utils import plot_model


def easy_drive():
    """
    Neural network approximates mapping between kinematics measures to throttle & brake signal

    batch input -  batch size * 2
    batch output - batch size * 2
    """
    model = Sequential()
    model.add(Dense(30, activation="relu",input_dim=2))
    #model.add(Dropout(0.8))
    #model.add(Dense(100, activation="relu"))
    #model.add(Dense(16, activation="relu"))
    model.add(Dense(30, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model

def create_model_neurons(neurons=10):
	# create model
    model = Sequential()
    model.add(Dense(neurons, activation="relu",input_dim=2))
    model.add(Dropout(0.2))
    model.add(Dense(2))
	# Compile model
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def test_model():
    model = Sequential()
    model.add(Flatten(input_shape=(67, 320)))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    return model

def drive_with_camera():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", input_shape=(67, 320, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model
