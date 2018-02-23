#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB), and the INTEL Visual Computing Lab.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function
import argparse
import logging
import random
import sys
import time
from Utils import utilities, nnet
from keras.wrappers.scikit_learn import KerasClassifier
from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line


def user_query():
    yes = {'yes','y', 'ye', ''}
    no = {'no','n'}
    print("Continue with training ?", " Please respond with 'yes' or 'no'")
    choice = input().lower()
    if choice in yes:
        return True
    elif choice in no:
        return False
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network to autonomously control throttle and brake of a car with given velocity and acceleration. Example syntax:\n\npython model.py -d udacity_dataset -m model.h5')
    parser.add_argument('--dataset-directory', '-d', dest='dataset_directory', type=str, required=True, help='Required string: Directory containing driving log and images.')
    parser.add_argument('--model-path', '-m', dest='model_path', type=str, required=False, default='model.h5', help='Required string: Name of model e.g model.h5.')
    parser.add_argument('--cpu-batch-size', '-c', dest='cpu_batch_size', type=int, required=False, default=100, help='Optional integer: Image batch size that fits in system RAM. Default 1000.')
    parser.add_argument('--gpu-batch-size', '-g', dest='gpu_batch_size', type=int, required=False, default=64, help='Optional integer: Image batch size that fits in VRAM. Default 512.')
    parser.add_argument('--randomize', '-r', dest='randomize', type=bool, required=False, default=False, help='Optional boolean: Randomize and overwrite driving log. Default False.')
    args = parser.parse_args()



    dataset_log = utilities.get_dataset_from_csv(args.dataset_directory)
    dataset_size = dataset_log.shape[0]
    validation_batch_size = int(0.2 * dataset_size)
    measurement_index = 0
    print("dataset size is",dataset_size)
    print("validation batch size is",validation_batch_size)
    validation_set = utilities.batch_preprocess(args.dataset_directory,
                                                measurement_range=(measurement_index,
                                                                measurement_index + validation_batch_size),
                                                debug=False)
    X_valid = validation_set['features']
    y_valid = validation_set['labels']
    measurement_index = validation_batch_size  # update measurement index to the end of the validation set
    model = nnet.easy_drive()  # initialize neural network model that will be iteratively trained in batches


    # raw_input returns the empty string for "enter"
    exit = False
    while (exit == False):
        response = user_query()
        if(response == True):
            print("start training")
            while measurement_index < dataset_size:
                    end_index = measurement_index + args.cpu_batch_size  # data taken for training based on cpu_batch_size
                    if end_index < dataset_size:
                        print("Pre-processing from index", measurement_index, "to index", end_index)
                        preprocessed_batch = utilities.batch_preprocess(args.dataset_directory, measurement_range=(measurement_index, end_index))
                    else:
                        print("Pre-processing from index", measurement_index, "to index", dataset_size)
                        preprocessed_batch = utilities.batch_preprocess(args.dataset_directory, measurement_range=(measurement_index, None))
                    X_batch = preprocessed_batch['features']
                    y_batch = preprocessed_batch['labels']
                    print("Done preprocessing.")
                    print("features data shape", X_batch.shape)
                    print("labels data shape", y_batch.shape)
                    model.fit(X_batch, y_batch, validation_data=(X_valid, y_valid), shuffle=True, nb_epoch=2000, batch_size=args.gpu_batch_size)
                    measurement_index += args.cpu_batch_size
            model.save(args.model_path)
            exit = True
        elif (response == False):
            exit = True
        else:
            pass
