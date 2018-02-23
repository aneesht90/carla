#!/usr/bin/env python3

"""Basic test run example."""

from __future__ import print_function


import argparse
import logging
import random
import sys
import time
import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import h5py
from PIL import Image
from flask import Flask
from io import BytesIO
from keras.models import load_model
from keras import __version__ as keras_version
from Utils import utilities, nnet



def main(model=None):
    _model = load_model(model)
    test_input = np.zeros((1,2))
    velocity = 5
    acceleration = 3
    test_input[0] = velocity , acceleration
    #test_input[1] = 6
    print("network input is: ",test_input)
    prediction = _model.predict(test_input, batch_size=1)
    brake       = [x[0] for x in prediction]
    throttle    = [x[1] for x in prediction]
    print("prediction is as follows: ","brake: ",brake ," throttle: ",throttle)



if __name__ == '__main__':

    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--model',
        type=str,
        default='model.h5',
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = argparser.parse_args()

    try:
        main(args.model)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
