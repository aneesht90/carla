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
    model.add(Dense(10, activation="relu",input_dim=2))
    model.add(Dense(10, activation="relu"))
    #model.add(Dropout(0.9))
    model.add(Dense(2))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model
