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

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import glob


def preprocess_color(image_matrix):
    image_matrix_cropped = image_matrix[70:137, 0:, :]
    image_matrix_cropped_normalized = image_matrix_cropped / 255 - 0.5
    return image_matrix_cropped_normalized


def preprocess_grayscale(image_matrix):
    image_matrix_gray = cv2.cvtColor(image_matrix, cv2.COLOR_RGB2GRAY)
    image_matrix_cropped = image_matrix_gray[70:137, 0:]
    image_matrix_cropped_normalized = image_matrix_cropped / 255 - 0.5
    return image_matrix_cropped_normalized


def preprocess_laplacian(image_matrix, debug=False):
    image_matrix_cropped = image_matrix[70:137, 0:, :]
    laplacian = np.empty_like(image_matrix_cropped)
    laplacian[:, :, 0] = np.absolute(cv2.Laplacian(image_matrix_cropped[:, :, 0], cv2.CV_64F))
    if debug:
        show_image((1, 1, 1), "laplacian 0", laplacian[:, :, 0], 1)
    laplacian[:, :, 1] = np.absolute(cv2.Laplacian(image_matrix_cropped[:, :, 1], cv2.CV_64F))
    if debug:
        show_image((1, 1, 1), "laplacian 0", laplacian[:, :, 1], 1)
    laplacian[:, :, 2] = np.absolute(cv2.Laplacian(image_matrix_cropped[:, :, 2], cv2.CV_64F))
    if debug:
        show_image((1, 1, 1), "laplacian 0", laplacian[:, :, 2], 1)
    laplacian_max = np.amax(laplacian, 2)
    laplacian_normalized = laplacian_max / (255) - 0.5
    return(laplacian_normalized)


def randomize_dataset_csv(csv_path):
    driving_log = pd.read_csv(csv_path, header=None)
    driving_log = driving_log.sample(frac=1).reset_index(drop=True)
    print("Overwriting CSV file: ", csv_path)
    driving_log.to_csv(csv_path, header=None, index=False)
    print("Done.")


def get_driving_log_path(image_input_dir):
    assert (os.path.exists(image_input_dir))
    log_file_list = glob.glob(os.path.join(image_input_dir, '*.csv'))
    #print("log list: ",log_file_list,"length: ",len(log_file_list))
    assert (len(log_file_list))
    log_path = log_file_list[0]
    return log_path



def get_dataset_from_csv(image_input_dir):
    assert (os.path.exists(image_input_dir))
    log_file_list = glob.glob(os.path.join(image_input_dir, '*.csv'))
    all_data = pd.DataFrame()
    for f in log_file_list:
        df = pd.read_csv(f, header=None)
        df.drop(df.index[:2], inplace=True)
        all_data = all_data.append(df,ignore_index=True)
    #print (all_data)
    return all_data


def get_dataset_from_pickle(pickle_file_path):
    with open(pickle_file_path, mode='rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data


def batch_preprocess_with_images(image_input_dir, l_r_correction=0.2, debug=False, measurement_range=None):
    """
    Preprocess all images and measurements then save them to disk in Keras-compatible format.
    # + numbers go right, - numbers go left. Thus for left camera we correct right and for right camera we collect left.
    """
    driving_log = get_dataset_from_csv(image_input_dir)
    if measurement_range[0]:
        measurement_index = measurement_range[0]
    else:
        measurement_index = 0
    if measurement_range[1]:
        max_measurement_index = measurement_range[1]
    else:
        max_measurement_index = driving_log.shape[0]
    assert(measurement_index < max_measurement_index)
    num_measurements = max_measurement_index - measurement_index

    num_images = num_measurements * 6  # * 6 because of left center and right image for each entry and their flipped versions.
    y_train = np.zeros(num_images)  # we 6X the number of measurements because we have 3 cameras and we flip each view to generate 6 (images, steering) pairs for each measurement
    X_train = np.zeros((num_images, 67, 320, 3))
    while measurement_index < max_measurement_index:
        datum_index = (measurement_index - measurement_range[0]) * 6
        # CENTER CAMERA IMAGE
        y_train[datum_index] = driving_log.iloc[measurement_index, 3]  # center image steering value added to dataset
        center_image_filename = driving_log.iloc[measurement_index, 0]
        center_image_path = os.path.join(image_input_dir, center_image_filename)
        if debug:
            print("Using center image path", center_image_path)
        center_image_matrix = cv2.imread(center_image_path)
        preprocessed_center_image_matrix = preprocess_color(center_image_matrix)
        X_train[datum_index, :, :, :] = preprocessed_center_image_matrix  # center image matrix added to dataset
        # LEFT CAMERA IMAGE
        y_train[datum_index + 1] = driving_log.iloc[measurement_index, 3] + l_r_correction  # left image steering value added to dataset
        left_image_filename = driving_log.iloc[measurement_index, 1]
        left_image_path = os.path.join(image_input_dir, left_image_filename)
        if debug:
            print("Using left image path", left_image_path)
        left_image_matrix = cv2.imread(left_image_path)
        preprocessed_left_image_matrix = preprocess_color(left_image_matrix)
        X_train[datum_index + 1, :, :, :] = preprocessed_left_image_matrix  # left image matrix added to dataset
        # RIGHT CAMERA IMAGE
        y_train[datum_index + 2] = driving_log.iloc[measurement_index, 3] - l_r_correction  # right image steering value added to dataset
        right_image_filename = driving_log.iloc[measurement_index, 2]
        right_image_path = os.path.join(image_input_dir, right_image_filename)
        if debug:
            print("Using right image path", right_image_path)
        right_image_matrix = cv2.imread(right_image_path)
        preprocessed_right_image_matrix = preprocess_color(right_image_matrix)
        X_train[datum_index + 2, :, :, :] = preprocessed_right_image_matrix  # right image matrix added to dataset
        # FLIPPED CENTER CAMERA IMAGE
        flipped_center = cv2.flip(preprocessed_center_image_matrix, flipCode=1)
        y_train[datum_index + 3] = y_train[datum_index]*-1
        X_train[datum_index + 3, :, :, :] = flipped_center
        # FLIPPED LEFT CAMERA IMAGE
        flipped_left = cv2.flip(preprocessed_left_image_matrix, flipCode=1)
        y_train[datum_index + 4] = y_train[datum_index + 1]*-1
        X_train[datum_index + 4, :, :, :] = flipped_left
        # FLIPPED RIGHT CAMERA IMAGE
        flipped_right = cv2.flip(preprocessed_right_image_matrix, flipCode=1)
        y_train[datum_index + 5] = y_train[datum_index + 2]*-1
        X_train[datum_index + 5, :, :, :] = flipped_right
        measurement_index += 1
        if debug:
            plt.figure(figsize=(15, 5))
            show_image((2, 3, 1), "Left View w/ Steering Angle " + str(y_train[datum_index]) + " Degrees", cv2.cvtColor(cv2.convertScaleAbs(preprocessed_left_image_matrix + 0.5, alpha=255), cv2.COLOR_BGR2RGB))
            show_image((2, 3, 2), "Center View w/ Steering Angle " + str(y_train[datum_index]) + " Degrees", cv2.cvtColor(cv2.convertScaleAbs(preprocessed_center_image_matrix + 0.5, alpha=255), cv2.COLOR_BGR2RGB))
            show_image((2, 3, 3), "Right View w/ Steering Angle " + str(y_train[datum_index + 2]) + " Degrees", cv2.cvtColor(cv2.convertScaleAbs(preprocessed_right_image_matrix + 0.5, alpha=255), cv2.COLOR_BGR2RGB))
            show_image((2, 3, 4), "Flipped Left View w/ Steering Angle " + str(y_train[datum_index + 4]) + " Degrees", cv2.cvtColor(cv2.convertScaleAbs(flipped_left + 0.5, alpha=255), cv2.COLOR_BGR2RGB))
            show_image((2, 3, 5), "Flipped Center View w/ Steering Angle " + str(y_train[datum_index + 3]) + " Degrees", cv2.cvtColor(cv2.convertScaleAbs(flipped_center + 0.5, alpha=255), cv2.COLOR_BGR2RGB))
            show_image((2, 3, 6), "Flipped Right View w/ Steering Angle " + str(y_train[datum_index + 5]) + " Degrees", cv2.cvtColor(cv2.convertScaleAbs(flipped_right + 0.5, alpha=255), cv2.COLOR_BGR2RGB))
            plt.show()
            plt.close()
        print('Pre-processed ', measurement_index, ' of ', max_measurement_index, ' measurements. Images:', center_image_filename, ' ', left_image_filename, ' ', right_image_filename)
    preprocessed_dataset = {'features': X_train, 'labels': y_train}
    return preprocessed_dataset


# note : to be implemented
def hyperparameter_search_model_selection(model_gen, X_train, y_train, param_grid):

    # clf = KerasClassifier(build_fn=model_gen,
    #                       epochs=50,
    #                       class_weight=class_weight,
    #                       verbose=0)

    clf = KerasClassifier(build_fn=model_gen,
                           verbose=0)


    grid = GridSearchCV(estimator=clf,
                        param_grid=param_grid,
                        n_jobs=1)
    #
    result = grid.fit(X_train, y_train)
    #
    # print("Best: %f using %s" % (result.best_score_, result.best_params_))
    # means = result.cv_results_['mean_test_score']
    # stds = result.cv_results_['std_test_score']
    # params = result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))

# note : to be implemented
def hyperparameter_search_over_model(model_gen, dataset_id, param_grid, cutoff=None,
                                     normalize_timeseries=False):

    X_train, y_train, _, _, is_timeseries = load_dataset_at(dataset_id,
                                                            normalize_timeseries=normalize_timeseries)
    max_nb_words, sequence_length = calculate_dataset_metrics(X_train)

    if sequence_length != MAX_SEQUENCE_LENGTH_LIST[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, sequence_length)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, _ = cutoff_sequence(X_train, None, choice, dataset_id, sequence_length)

    if not is_timeseries:
        print("Model hyper parameters can only be searched for time series models")
        return

    classes = np.unique(y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    recip_freq = len(y_train) / (len(le.classes_) *
                                 np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]

    y_train = to_categorical(y_train, len(np.unique(y_train)))

    clf = KerasClassifier(build_fn=model_gen,
                          epochs=50,
                          class_weight=class_weight,
                          verbose=0)

    grid = GridSearchCV(clf, param_grid=param_grid,
                        n_jobs=1, verbose=10, cv=3)

    result = grid.fit(X_train, y_train)

    print("Best: %f using %s" % (result.best_score_, result.best_params_))
    means = result.cv_results_['mean_test_score']
    stds = result.cv_results_['std_test_score']
    params = result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def batch_preprocess(image_input_dir, l_r_correction=0.2, debug=False, measurement_range=None):
    """
    Preprocess all images and measurements then save them to disk in Keras-compatible format.
    # + numbers go right, - numbers go left. Thus for left camera we correct right and for right camera we collect left.
    """
    driving_log = get_dataset_from_csv(image_input_dir)
    if measurement_range[0]:
        measurement_index = measurement_range[0]
    else:
        measurement_index = 1
    if measurement_range[1]:
        max_measurement_index = measurement_range[1]
    else:
        max_measurement_index = driving_log.shape[0]
    assert(measurement_index < max_measurement_index)
    num_measurements = max_measurement_index - measurement_index

    y_train = np.zeros((num_measurements,2),dtype=np.float32)
    x_train = np.zeros((num_measurements,2),dtype=np.float32)
    print("preprocessing the data. ","number of samples: ",max_measurement_index)
    while measurement_index < max_measurement_index-1:
        datum_index = (measurement_index - measurement_range[0])
        y_train[datum_index,0] = driving_log.iloc[measurement_index, 2] # brake
        y_train[datum_index,1] = driving_log.iloc[measurement_index, 3] # throttle
        x_train[datum_index,0] = driving_log.iloc[measurement_index, 4] # velocity
        x_train[datum_index,1] = driving_log.iloc[measurement_index, 5] # acceleration - y
        measurement_index += 1
        #print("iter:", x_train)
    preprocessed_dataset = {'features': x_train, 'labels': y_train}
    return preprocessed_dataset


def save_dict_to_pickle(dataset, file_path):
    print("Saving data to", file_path, "...")
    pickle.dump(dataset, open(file_path, "wb"), protocol=4)  # protocol=4 allows file sizes > 4GB
    print("Done.")


def show_image(location, title, img, width=3, open_new_window=False):
    if open_new_window:
        plt.figure(figsize=(width, width))
    plt.subplot(*location)
    plt.title(title, fontsize=8)
    plt.axis('off')
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    if open_new_window:
        plt.show()
        plt.close()
