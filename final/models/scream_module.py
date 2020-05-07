from __future__ import print_function
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras import models
from keras import layers
import numpy as np
import pandas as pd
import sklearn
# import freesound
from audioread import NoBackendError
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import glob, os
from pathlib import Path
import csv
import warnings  # record warnings from librosa
from sklearn.model_selection import train_test_split
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.activations import relu, sigmoid
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn.preprocessing import StandardScaler
from module_data import *
import json
import yaml
from keras.models import model_from_json


def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """

    path = Path('module_data')
    current_path_x_train = path / f"scream_train_x.pkl"
    with current_path_x_train.open('rb') as file:
        X_train = pkl.load(file)

    current_path_x_test = path / f"scream_test_x.pkl"
    with current_path_x_test.open('rb') as file:
        X_test = pkl.load(file)

    current_path_y_train = path / f"scream_train_y.pkl"
    with current_path_y_train.open('rb') as file:
        Y_train = pkl.load(file)

    current_path_y_test = path / f"scream_test_y.pkl"
    with current_path_y_test.open('rb') as file:
        Y_test = pkl.load(file)
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = data()
    # keras
    #  how i expect to receive a model:
    # loaded model = jsonLoad \pandaLoad mori's choice on the saving format
    model = models.Sequential()

    # print(X_train_kfold_scaled.shape[1])  #  45 (is the number of columns for each sample)
    model.add(layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # the 1 means binary classification

    model.compile(optimizer='adam'
                  , loss='binary_crossentropy'
                  , metrics=['accuracy'])
    # save as JSON without being trained
    # json_string = model.to_json()
    # with open('scream_model.json', 'w') as f:
    #     json.dump(json_string, f, ensure_ascii=False)


    # model reconstruction from JSON:
    # path = Path('models/module_data')
    # current_model = path / f"scream_model.json"
    # with open('scream_model.json') as file:
    #     model_data = json.load(file)
    # model = model_from_json(model_data)

