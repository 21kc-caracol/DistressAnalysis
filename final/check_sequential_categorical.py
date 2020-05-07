import pandas as pd
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score

import numpy as np
from evaluate_genericClf import global_For_Clf


# FUNCTIONS


def baseline_model():
    model = Sequential()
    model.add(layers.Dense(32, activation='relu', input_dim=clfGlobals.getInputDim()))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(11, activation='softmax'))   # total of 11 classes including NearMiss...
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



if __name__ == "__main__":
    # todo add user input- basic clf for mori-use result
    clfGlobals = global_For_Clf('scream')  # create global variable
    # LOAD
    csv_path = f'csv/whisper/data_experiment_2_mfcc_20.csv'

    # load csv
    data_csv = pd.read_csv(csv_path)
    data_no_fileName = data_csv.drop(['filename'], axis=1)
    only_features = np.array(data_no_fileName.iloc[:, :-1], dtype=float)
    # print(only_features.shape)  # 26 is correct
    # normalize
    scaler_filenPath = f'saved_scalers/scaler_mfcc_20.save'
    scaler = joblib.load(scaler_filenPath)
    X = scaler.transform(only_features)


    # encode Y
    Y = data_no_fileName.iloc[:, -1]  # only labels
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    # print(dummy_y)

    estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=11, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



