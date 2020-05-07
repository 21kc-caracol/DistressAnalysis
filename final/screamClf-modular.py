#  imports
import sys
import librosa
import librosa.display

import matplotlib.pyplot as plt

from keras import models
from keras import layers
import numpy as np
import pandas as pd

import sklearn

import freesound

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


#  global objects
#todo add private/public inside class
class global_For_Clf():
    def __init__(self):
        self.n_mfcc = 40  # lev's initial value here was 40- this is the feature resolution- usually between 12-40
        self.k_folds = 5  # amount of folds in k-fold
        # inside create_csv() more columns will be added to the csv head
        # TODO lev-future_improvement edit/add to get better results
        self.csv_initial_head = 'filename spectral_centroid zero_crossings spectral_rolloff chroma_stft rms mel_spec'

        self.data_file_path= 'dataTestingScream.csv'
        self.min_wav_duration = 0.5  # wont use shorter wav files
        self.clf_label = 'scream'
        self.nearMissRatio = 2  # 2 means <positives amount>/2
        #                           which means were taking 50% from nearMiss_<clf label> for negatives

        self.nearMissLabel= "NearMiss_"+str(self.clf_label)
        self.csv_to_pkl_path= "pickle/scream/combined_lower_amount.pkl"
        self.path_csv_train_test_data = "csv/scream/train_test_data.csv"  # chosen 1:1 ratio data, selected from data.csv

        self.Kfold_testSize= 0.2

    def getInputDim(self):
        amount = len(self.csv_initial_head.split()) + self.n_mfcc - 1 # -1 because filename isnt a feature
        return amount

#  exceptions
class NotEnoughPositiveSamples(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def extract_feature_to_csv(wav_path, label, data_file_path, min_wav_duration, fcc_amount):
    """

    :return: writes one row to wav_path with extracted features

    """
    # extract features for a wav file
    wav_name = wav_path.name  # 110142__ryding__scary-scream-4.wav
    wav_name = wav_name.replace(" ", "_")  # lev bug fix to allign csv columns
    #  print(wav_name)

    """
    # lev upgrading error tracking- know which file caused the error
    try:
    """
    wav_data, sampling_rate = librosa.load(wav_path, duration=5)

    wav_duration = librosa.get_duration(y=wav_data, sr=sampling_rate)

    # lev- dont use really short audio
    if (wav_duration < min_wav_duration):
        print("skipping " + wav_name + " ,duration= " + str(wav_duration))
        return

    with warnings.catch_warnings(record=True) as feature_warnings:
        #  spectral_centroid
        feature_wav_spec_cent = librosa.feature.spectral_centroid(y=wav_data, sr=sampling_rate)
        #  print(feature_wav_spec_cent.shape)  #  (1, 216)

        #  zero crossings
        zcr = librosa.feature.zero_crossing_rate(wav_data)
        #  print("sum "+ str(np.sum(zcr)))

        #  spectral_rolloff
        rolloff = librosa.feature.spectral_rolloff(y=wav_data, sr=sampling_rate)
        # print(rolloff.shape)
        # print(rolloff[0][0:3])

        #  chroma_stft
        chroma_stft = librosa.feature.chroma_stft(y=wav_data, sr=sampling_rate)
        #  print(chroma_stft.shape)

        #  rms and mfccs
        n_mfcc = fcc_amount  # resolution amount
        mfccs = librosa.feature.mfcc(y=wav_data, sr=sampling_rate, n_mfcc=n_mfcc)
        S, phase = librosa.magphase(mfccs)
        rms = librosa.feature.rms(S=S)
        #  print(rms.shape)

        # mel spectogram
        mel_spec= librosa.feature.melspectrogram(y=wav_data, sr=sampling_rate)


        # mfccs
        #  print(mfccs.shape)
        # if there ara warnings- print and continue- for example Warning: Trying to estimate tuning from empty frequency set
        # this is an OK warning- it just means that its really quiet..as in street ambient during the evenning..its a
        # good negative example.
        if len(feature_warnings) > 0:
            for feature_warning in feature_warnings:
                print("Warning: {} Triggered in:\n {}\nwith a duration of {} seconds.\n".format(
                    feature_warning.message, wav_path, wav_duration))

        # got here - no warnings for this wav_path
        # normalize what isnt normalized
        to_append = f'{wav_name} {np.mean(feature_wav_spec_cent)} {np.mean(zcr)} {np.mean(rolloff)} {np.mean(chroma_stft)}' \
                    f' {np.mean(rms)} {np.mean(mel_spec)}'
        for e in mfccs:
            to_append += f' {np.mean(e)}'

        to_append += f' {label}'

        #  save to csv (append new lines)
        file = open(data_file_path, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

        #  print(to_append)


def create_csv():
    """
    input: uses screamGlobals for input
    output: .csv file with screamGlobals.csv_initial_head columns
    """
    # important variables
    data_file_path = screamGlobals.data_file_path
    min_wav_duration = screamGlobals.min_wav_duration
    #  print(data_file_path, min_wav_duration)

    #  prevent data file over run by accident
    if os.path.exists(data_file_path):
        text = input(f'Press the space bar to override {data_file_path} and continue with the script')
        if text != ' ':
            sys.exit('User aborted script, data file saved :)')

    # create header for csv
    header = screamGlobals.csv_initial_head
    fcc_amount = screamGlobals.n_mfcc
    for i in range(1, fcc_amount + 1):
        header += f' mfcc_{i}'
    header += ' label'
    header = header.split()  # split by spaces as default

    file = open(data_file_path, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    # load features from each wav file- put inside the lines below as a function

    # reaching each wav file
    path_train = Path("train")
    for path_label in sorted(path_train.iterdir()):
        print("currently in : " + str(path_label))  # train\negative
        positiveOrNegative = path_label.name  # negative
        #  print(label)
        for path_class in tqdm(sorted(path_label.iterdir())):
            # print info
            print("currently in class: " + str(path_class))
            # print amount of files in directory
            onlyfiles = next(os.walk(path_class))[2]  # dir is your directory path as string
            wav_amount: int = len(onlyfiles)
            print("wav amount= " + str(wav_amount))
            #  true_class= path_class.name
            #  print(true_class)
            #  print(path_class)  #  train\negative\scream
            #  print("name: "+ str(path_class.name))

            # lev improvement according to coordination with mori- irrelevant since 7.8.19
            if (positiveOrNegative == "positive"):
                label = path_class.name  # scream
            else:
                """
                lev- updating to differentiate near misses and far misses.
                keeping if-else structure for future options

                old:
                print(f"switching label from {path_class.name} to <negative>")  # added reporting
                label = "negative"
                new:

                """
                label = path_class.name  # NearMiss_scream

            wave_file_paths = path_class.glob('**/*.wav')  # <class 'generator'>
            #  print(type(wave_file_paths))
            count = 0  # for progress tracking
            print('covered WAV files: ')
            for wav_path in sorted(wave_file_paths):
                count += 1
                if (count % 50) == 0:
                    fp = sys.stdout
                    print(str(count), end=' ')
                    fp.flush()  # makes print flush its buffer (doesnt print without it)
                #  print(type(wav_path))  #  <class 'pathlib.WindowsPath'>
                #  print(wav_path)  #  train\positive\scream\110142__ryding__scary-scream-4.wav
                #  print(wav_path.name)  #  110142__ryding__scary-scream-4.wav
                try:
                    #  keeping as parameters data_file_path, min_wav_duration even though its in screamGlobals
                    #  in order to emphasis its an inner function of create_csv()
                    extract_feature_to_csv(wav_path, label, data_file_path, min_wav_duration, fcc_amount)
                except NoBackendError as e:
                    print("audioread.NoBackendError " + "for wav path " + str(wav_path))
                    continue  # one file didnt work, continue to next one


def create_lower_bound_data_pickle_and_csv(csv_path, label):
    """
    note(lev): because usually we will have more negatives than positives then this function
        chooses randomly the negatives samples so that it will have 1:1 ratio with the true label
        and within the amount of false labels, it Stratifies to keep the same ratio of
        Near Misses for both train and test data.
        (this has proven to increase the k-fold average accuracy from 0.45 to 0.85

    """

    print(f'choosing max samples randomly while preserving 1:1 ratio for {label}:<all the rest as one group>')
    # use Pandas package for reading csv
    data_csv = pd.read_csv(csv_path)
    # print(data_csv[data_csv.label == 'scream'])  #  [367 rows x 47 columns]
    # print(len(data_csv[data_csv.label == 'scream']))  # 367

    # find lower amount from types of labels

    pos_amount = len(data_csv[data_csv.label == label])
    neg_amount = len(data_csv[data_csv.label != label])
    print("positives: " + str(pos_amount) + " negatives: " + str(neg_amount))
    lower_amount = min(pos_amount, neg_amount)
    print("lower bound: " + str(lower_amount))

    # take Max of 50% from NearMiss_<clf label> and then choose randomly from the rest of negatives
    nearMissMaxAmount = pos_amount // screamGlobals.nearMissRatio

    data_csv_negatives_nearMiss = data_csv.loc[data_csv.label == screamGlobals.nearMissLabel,:]  # take all valid rows
    nearMissActualAmount = len(data_csv_negatives_nearMiss)
    NearMissAmountToTake = nearMissActualAmount if nearMissActualAmount < nearMissMaxAmount else nearMissMaxAmount
    # take near misses for this classifier
    data_csv_negatives_NearMiss = data_csv_negatives_nearMiss.sample(n=NearMissAmountToTake)

    # take random negatives that aren't near miss
    negatives_amount_left_to_take= lower_amount - NearMissAmountToTake
    rest_of_negatives= data_csv.loc[~data_csv['label'].isin([label,screamGlobals.nearMissLabel])]  # take all valid rows

    negatives_lower_amount_samples = data_csv_negatives_NearMiss.append(rest_of_negatives.sample(n= negatives_amount_left_to_take))
    assert (len(negatives_lower_amount_samples) == lower_amount)

    #  positives - taking random rows
    data_csv_positives = data_csv[data_csv.label == label]

    # create pandas dataframe with lower_amount rows randomly
    positives_lower_amount_samples = data_csv_positives.sample(n=lower_amount)

    # combine
    combined_lower_amount = positives_lower_amount_samples
    # have to assign, returns appended datadrame
    combined_lower_amount = combined_lower_amount.append(negatives_lower_amount_samples)
    # print(len(combined_lower_amount))  # 734 ,when lower bound: 367


    # saving pandas dataframe to csv - for data analysis purposes
    #  TODO lev future - maybe build a function- you already copied this logic 3 times
    if os.path.exists(screamGlobals.path_csv_train_test_data):
        text = input(f'Press the space bar to override {screamGlobals.path_csv_train_test_data} and continue with the script')
        if text != ' ':
            sys.exit('User aborted script, pickle file saved :)')
    combined_lower_amount.to_csv(screamGlobals.path_csv_train_test_data)

    #TODO RETURN HERE LINES OF CODE FOR EDITING LABELS


    assert (len(combined_lower_amount) == lower_amount*2)
    # saving pandas dataframe to pickle - modularity
    #  prevent pickle file over run by accident
    if os.path.exists(screamGlobals.csv_to_pkl_path):
        text = input(f'Press the space bar to override {screamGlobals.csv_to_pkl_path} and continue with the script')
        if text != ' ':
            sys.exit('User aborted script, pickle file saved :)')
    combined_lower_amount.to_pickle(screamGlobals.csv_to_pkl_path)



def splittingData(k, pkl_path):
    """"
    arrange data for split, then split into test and k-fold
    """
    print(f' splittingData: saving (k-fold and Test) groups as pickle files')
    # load combined dataset as pandas dataframe (50%-50% ratio between labels)
    combined_lower_amount = pd.read_pickle(pkl_path)

    # prepare dataFrame for stratified and for split
    data_no_fileName = combined_lower_amount.drop(['filename'], axis=1)
    #  print(data_no_fileName.shape)  # ( , ) with label column

    # encode strings of labels to integers
    labels_list = data_no_fileName.iloc[:, -1]
    # print(labels_list.shape)  # (734,)
    encoder = LabelEncoder()
    encoded_labels_csv = np.array(encoder.fit_transform(labels_list))
    # print(encoded_labels_csv)  #  [0 0 0 ... 1 1 1]

    # print(encoded_labels_csv.shape) # (734,)

    # print(list(encoder.inverse_transform([0,1])))  #  ['negative', 'scream']

    # take all except labels column. important to add the dtype=float, otherwise im getting an error in the kfold.
    only_features = np.array(data_no_fileName.iloc[:, :-1], dtype=float)
    #  print(only_features.shape)  # ( , )

    """"
    splitting stages
    """
    # split for test and k-fold
    X_for_k_fold, X_test, y_for_k_fold, y_test = train_test_split\
        (only_features, encoded_labels_csv, test_size=screamGlobals.Kfold_testSize , stratify=encoded_labels_csv)
    # print(len(y_for_k_fold))  # 587
    # print(len(y_test))  # 147

    # after stratify- lets keep only binary classification (merging Near Miss and Far miss into one label
    print("merging all negative labels into one label")

    # find the positive label transformation
    clf_label_int= encoder.transform([screamGlobals.clf_label])[0]  # 0 because it returns a list
    #  print(clf_label_int)  # 2

    #give temp values in order to prevent conflicts
    tempIntLabel :int= 111
    IntNegative :int = 0
    IntPositive :int= 1
    y_for_k_fold[y_for_k_fold == clf_label_int] = tempIntLabel
    y_test[y_test == clf_label_int] = tempIntLabel

    y_for_k_fold[y_for_k_fold != tempIntLabel] = IntNegative
    y_test[y_test != tempIntLabel] = IntNegative

    # switch to binary
    y_for_k_fold[y_for_k_fold == tempIntLabel] = IntPositive
    y_test[y_test == tempIntLabel] = IntPositive

    # safely handle existing pickle files
    if os.path.exists('pickle/scream/X_test.pkl'):
        text = input(f'Press the space bar once and then press Enter to override all pickle files and continue with the script')
        if text != ' ':
            sys.exit('User aborted script, pickle files saved :)')
    # save test data as pkl
    path_x_test = Path("pickle/scream/X_test.pkl")
    with path_x_test.open('wb') as file:
        pkl.dump(X_test, file)
    path_y_test = Path("pickle/scream/y_test.pkl")
    with path_y_test.open('wb') as file:
        pkl.dump(y_test, file)

    # stratified split
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    skf.get_n_splits(X_for_k_fold, y_for_k_fold)  # StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    # print(skf)

    fold_num = 1  # use for saving pkl files with different names
    path = Path('pickle/scream/folds')
    for train_index, test_index in skf.split(X_for_k_fold, y_for_k_fold):
        X_train_kfold, X_test_kfold = X_for_k_fold[train_index], X_for_k_fold[test_index]
        y_train_kfold, y_test_kfold = y_for_k_fold[train_index], y_for_k_fold[test_index]
        """
        lev- verified the expected split ratio in function how_to_stratified_k_fold  

        unique, counts = np.unique(y_train_kfold, return_counts=True)
        print(dict(zip(unique, counts)))  # b4 split for stratford and test {0: 293, 1: 293} | after {0: 235, 1: 234}
        unique, counts = np.unique(y_test_kfold, return_counts=True)
        print(dict(zip(unique, counts)))  # b4 split for stratford and test {0: 74, 1: 74} | after {0: 59, 1: 59}

        """
        # next lines are duplicates , dont want to copy to another function to save computation time
        current_path_x_train = path / f"X_train_kfold_{fold_num}.pkl"

        # print(current_path)  # pickle\folds\fold_x_train_1.pkl
        with current_path_x_train.open('wb') as file:
            pkl.dump(X_train_kfold, file)

        current_path_x_test = path / f"X_test_kfold_{fold_num}.pkl"
        with current_path_x_test.open('wb') as file:
            pkl.dump(X_test_kfold, file)

        current_path_y_train = path / f"y_train_kfold_{fold_num}.pkl"
        with current_path_y_train.open('wb') as file:
            pkl.dump(y_train_kfold, file)

        current_path_y_test = path / f"y_test_kfold_{fold_num}.pkl"
        with current_path_y_test.open('wb') as file:
            pkl.dump(y_test_kfold, file)

        fold_num += 1


def normalize_builtClassifier(k):
    """"
    important to use scaling only after splitting the data into train/validation/test
    scale on training set only, then use the returned "fit" parameters to scale validation and test
    """
    # load from pickle test data
    path_x_test = Path("pickle/scream/X_test.pkl")
    with path_x_test.open('rb') as file:
        X_test_loaded = pkl.load(file)
    path_y_test = Path("pickle/scream/y_test.pkl")
    with path_y_test.open('rb') as file:
        y_test_loaded = pkl.load(file)

    # print(X_test_loaded.shape)  # (147, 45)
    # print(y_test_loaded.shape)  # (147,)

    # built loop from 1-k including upper bound...first of all load pickle, then normalize, then send to keras...
    path = Path('pickle/scream/folds')
    for fold in range(1, k + 1):
        current_path_x_train = path / f"X_train_kfold_{fold}.pkl"
        with current_path_x_train.open('rb') as file:
            X_train_kfold = pkl.load(file)

        current_path_x_test = path / f"X_test_kfold_{fold}.pkl"
        with current_path_x_test.open('rb') as file:
            X_test_kfold = pkl.load(file)

        current_path_y_train = path / f"y_train_kfold_{fold}.pkl"
        with current_path_y_train.open('rb') as file:
            y_train_kfold = pkl.load(file)

        current_path_y_test = path / f"y_test_kfold_{fold}.pkl"
        with current_path_y_test.open('rb') as file:
            y_test_kfold = pkl.load(file)

        # print(X_train_kfold.shape,X_test_kfold.shape,y_train_kfold.shape,y_test_kfold.shape)
        # (469, 45) (118, 45) (469,) (118,)

        # scale data
        scaler = StandardScaler()
        scaler.fit(X_train_kfold)  # must call fit before calling transform.fitting on train, using on train+test+valid
        X_train_kfold_scaled = scaler.transform(X_train_kfold)
        # print(np.amax(X_train_kfold))  # 9490.310668945312
        # print(np.amax(X_train_kfold_scaled))  # 8.236592246485245
        X_test_kfold_scaled = scaler.transform(X_test_kfold)
        X_test_scaled = scaler.transform(X_test_loaded)

        # keras
        # here mori will give me a pre-trained model

        model = models.Sequential()

        # print(X_train_kfold_scaled.shape[1])  #  45 (is the number of columns for each sample)
        #model.add(layers.Dense(256, activation='relu', input_shape=(X_train_kfold_scaled.shape[1],)))
        model.add(layers.Dense(256, activation='relu', input_dim=screamGlobals.getInputDim()))
        model.add(layers.Dense(128, activation='relu'))

        model.add(layers.Dense(64, activation='relu'))

        model.add(layers.Dense(1, activation='sigmoid'))  # the 1 means binary classification

        model.compile(optimizer='adam'
                      , loss='binary_crossentropy'
                      , metrics=['accuracy'])

        # train model on training set of Kfold
        history = model.fit(X_train_kfold_scaled,
                            y_train_kfold,
                            epochs=20,
                            batch_size=128)

        test_loss, test_acc = model.evaluate(X_test_kfold_scaled, y_test_kfold)

        print(f'test_acc in fold number {fold}: ', test_acc)

        results = model.evaluate(X_test_scaled, y_test_loaded)
        print(f'results on the test data in fold number {fold}: ', results)




if __name__ == "__main__":
    print("executing screamClf flow")
    # change global definitions at the top of the file inside global_For_Clf class #todo lev maybe create config file
    screamGlobals = global_For_Clf()
    #create_csv()
    create_lower_bound_data_pickle_and_csv(screamGlobals.data_file_path, screamGlobals.clf_label)
    splittingData(screamGlobals.k_folds, screamGlobals.csv_to_pkl_path)
    normalize_builtClassifier(screamGlobals.k_folds)



