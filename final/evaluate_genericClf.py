#  imports
import sys
import librosa
import librosa.display

import matplotlib.pyplot as plt
from id3 import Id3Estimator

from keras import models
from keras import layers
import numpy as np
import pandas as pd

import sklearn

from audioread import NoBackendError
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.models import load_model

from tqdm import tqdm
import os

from pathlib import Path
import csv
import warnings  # record warnings from librosa
from sklearn.model_selection import train_test_split
import pickle as pkl

import json
from keras.models import model_from_json

# from models import get_optimised_model
from models import get_optimised_model_final

import datetime

#  global objects
# todo add private/public inside class
class global_For_Clf():
    def __init__(self, clf_label):
        #  changed for every class (for example: scream, cry, ...)
        self.clf_label = clf_label  # have to create a clf with a label

        # keeping the hardcoded 20 mfcc below until end of project submission, later update it to generic mfcc amount
        self.data_file_path = 'csv/'+str(self.get_clf_label())+'/data_'+str(self.get_clf_label())+'_mfcc_20.csv'  # cry
        self.csv_to_pkl_path = 'pickle/'+str(self.get_clf_label())+'/combined_lower_amount.pkl' # relevant to modular file TODO currently this is only for scream
        self.path_csv_train_test_data = 'csv/'+str(self.get_clf_label())+'/train_test_data.csv'  # chosen 1:1 ratio data, selected from data.csv
        self.resultsPath = 'results/'+str(self.get_clf_label())+'/experiments_results.csv'

        # end of class changes

        self.n_mfcc = 20  # lev's initial value here was 40- this is the feature resolution- usually between 12-40
        self.k_folds = 5  # amount of folds in k-fold
        # inside create_csv() more columns will be added to the csv head
        # TODO lev-future_improvement edit/add to get better results
        self.csv_initial_head = 'filename spectral_centroid zero_crossings spectral_rolloff chroma_stft rms mel_spec'

        self.min_wav_duration = 0.5  # wont use shorter wav files

        self.nearMissRatio = 2  # 2 means <positives amount>/2
        #                           which means were taking 50% from nearMiss_<clf label> for negatives

        self.nearMiss_samples = -1  # -1 is initial invalid value which will be changed on relevant functions
        self.nearMissLabel = "NearMiss_" + str(self.clf_label)

        self.Kfold_testSize = 0.2

        self.sampling_data_repetitions = 5  # sampling randomly the data to create 1:1 ratio
        self.k_fold_repetitions: int = 5  # doing repeated k-fold for better evaluation

        self.positives = -1  # -1 represents invalid value as initial value
        self.negatives = -1

        self.try_lower_amount = np.inf

        self.model = None  # here a model will be saved- the saved model shouldn't be trained
        self.finalModelsPath = 'models/final_models'
        self.isTrained = False

        self.userInput = ''

    def getInputDim(self):
        amount = len(self.csv_initial_head.split()) + self.n_mfcc - 1  # -1 because filename isnt a feature
        return amount

    def get_total_samples(self):
        return self.positives + self.negatives

    def get_model_name(self):
        model_name = (type(self.model)).__name__
        return model_name

    def get_clf_label(self):
            return self.clf_label


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
    wav_name = wav_name.replace(" ", "_")  # lev bug fix to align csv columns

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
        mel_spec = librosa.feature.melspectrogram(y=wav_data, sr=sampling_rate)

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
    data_file_path = clfGlobals.data_file_path
    min_wav_duration = clfGlobals.min_wav_duration
    #  print(data_file_path, min_wav_duration)
    """
    #  prevent data file over run by accident
    if os.path.exists(data_file_path):
        text = input(f'Press the space bar to override {data_file_path} and continue with the script')
        if text != ' ':
            sys.exit('User aborted script, data file saved :)')
    """
    if os.path.exists(data_file_path):
        # verify table fits the mfcc number- if True- return (continue with script as usuall), else- raise Error
        n_mfcc_number = clfGlobals.n_mfcc
        with open(data_file_path) as csvFile:
            reader = csv.reader(csvFile)
            field_names_list = next(reader)  # read first row only (header)
            mfcc_list = [x for x in field_names_list if x.startswith("mfcc")]
            len_actual_mfcc_features = len(mfcc_list)
        if len_actual_mfcc_features == n_mfcc_number:
            print(f'OK: {len_actual_mfcc_features} ==  n_mfcc_number={n_mfcc_number}')
            return
        else:
            raise Exception(f'len_actual_mfcc_features'
                            f'(mfcc inside {data_file_path}={len_actual_mfcc_features},'
                            f' but n_mfcc_number(inside globals class of this script)={n_mfcc_number},'
                            f' values must be equal.')

    # create header for csv
    header = clfGlobals.csv_initial_head
    fcc_amount = clfGlobals.n_mfcc
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
                wav_path = Path(wav_path)
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


def create_lower_bound_data_panda(csv_path, label):
    """
    note(lev): because usually we will have more negatives than positives then this function
        chooses randomly the negatives samples so that it will have 1:1 ratio with the true label
        and within the amount of false labels, it Stratifies to keep the same ratio of
        Near Misses for both train and test data.
        (this has proven to increase the k-fold average accuracy from 0.45 to 0.85

    if supplied a lower_bound which is lower than the negatives or positives amount it will
    act as above but with |"lowe bound"| positives and |"lower bound"| negatives
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
    lower_amount = min(pos_amount, neg_amount, clfGlobals.try_lower_amount)
    print("lower bound: " + str(lower_amount))

    # take Max of 50% from NearMiss_<clf label> and then choose randomly from the rest of negatives
    # lev bug fix should take lower amount as the Numerator
    #nearMissMaxAmount = pos_amount // screamGlobals.nearMissRatio
    nearMissMaxAmount = lower_amount // clfGlobals.nearMissRatio
    #  print("near miss max amount: ",nearMissMaxAmount)

    data_csv_negatives_nearMiss = data_csv.loc[data_csv.label == clfGlobals.nearMissLabel, :]  # take all valid rows
    nearMissActualAmount = len(data_csv_negatives_nearMiss)
    NearMissAmountToTake = nearMissActualAmount if nearMissActualAmount < nearMissMaxAmount else nearMissMaxAmount
    clfGlobals.nearMiss_samples = NearMissAmountToTake
    print(f"take {NearMissAmountToTake} near misses")
    # take near misses for this classifier
    data_csv_negatives_NearMiss = data_csv_negatives_nearMiss.sample(n=NearMissAmountToTake)

    # take random negatives that aren't near miss
    negatives_amount_left_to_take = lower_amount - NearMissAmountToTake
    #lev- bug fix: assert should be valid if left expression also "equals 0"
    assert (negatives_amount_left_to_take >= 0)
    rest_of_negatives = data_csv.loc[
        ~data_csv['label'].isin([label, clfGlobals.nearMissLabel])]  # take all valid rows

    negatives_lower_amount_samples = data_csv_negatives_NearMiss.append(
        rest_of_negatives.sample(n=negatives_amount_left_to_take))
    assert (len(negatives_lower_amount_samples) == lower_amount)
    # prepare for results tracking
    clfGlobals.positives = lower_amount
    clfGlobals.negatives = lower_amount

    #  positives - taking random rows
    data_csv_positives = data_csv[data_csv.label == label]

    # create pandas dataframe with lower_amount rows randomly
    positives_lower_amount_samples = data_csv_positives.sample(n=lower_amount)

    # combine
    combined_lower_amount = positives_lower_amount_samples
    # have to assign, returns appended datadrame
    combined_lower_amount = combined_lower_amount.append(negatives_lower_amount_samples)
    # print(len(combined_lower_amount))  # 734 ,when lower bound: 367

    """"
    dont need  safe override and data analysis in evaluation process
    # saving pandas dataframe to csv - for data analysis purposes
    #  TODO lev future - maybe build a function- you already copied this logic 3 times
    if os.path.exists(screamGlobals.path_csv_train_test_data):
        text = input(f'Press the space bar to override {screamGlobals.path_csv_train_test_data} and continue with the script')
        if text != ' ':
            sys.exit('User aborted script, pickle file saved :)')
    combined_lower_amount.to_csv(screamGlobals.path_csv_train_test_data)

    #TODO RETURN HERE LINES OF CODE FOR EDITING LABELS 

    # saving pandas dataframe to pickle - modularity
    #  prevent pickle file over run by accident
    if os.path.exists(screamGlobals.csv_to_pkl_path):
        text = input(f'Press the space bar to override {screamGlobals.csv_to_pkl_path} and continue with the script')
        if text != ' ':
            sys.exit('User aborted script, pickle file saved :)')
    combined_lower_amount.to_pickle(screamGlobals.csv_to_pkl_path)
    """

    assert (len(combined_lower_amount) == lower_amount * 2)
    return combined_lower_amount


def create_default_sequential_for_mfcc_size():
    """
    created this func for experiment2 due to the need of a default classifier
    """

    model = models.Sequential()

    # model.add(layers.Dense(256, activation='relu', input_shape=(X_train_kfold_scaled.shape[1],)))
    model.add(layers.Dense(32, activation='relu', input_dim=clfGlobals.getInputDim()))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # the 1 means binary classification

    model.compile(optimizer='adam'
                  , loss='binary_crossentropy'
                  , metrics=['accuracy'])
    clfGlobals.model = model
    clfGlobals.isTrained = False
    return model


def get_stratified_results(k, X_for_k_fold, y_for_k_fold, IntPositive, clfGlobals):
    """
    param: IntPositive- the integer representing our true classifiers label, for example: for scream classifier
        if scream class is represented by 1 ==> so  IntPositive = 1 . became necessary when wanted to return f1-score
        for this class only.
    """
    print("get_stratified_results() in process")
    # stratified split
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    skf.get_n_splits(X_for_k_fold, y_for_k_fold)  # StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    # print(skf)

    scores_accuracy_k_fold = []
    scores_f1_k_fold = []
    scores_recall_fold = []
    for train_index, test_index in skf.split(X_for_k_fold, y_for_k_fold):
        X_train_kfold, X_test_kfold = X_for_k_fold[train_index], X_for_k_fold[test_index]
        y_train_kfold, y_test_kfold = y_for_k_fold[train_index], y_for_k_fold[test_index]
        """"
        important to use scaling only after splitting the data into train/validation/test
        scale on training set only, then use the returned "fit" parameters to scale validation and test
        """
        # scale data
        scaler = StandardScaler()
        scaler.fit(X_train_kfold)  # must call fit before calling transform.fitting on train, using on train+test+valid
        X_train_kfold_scaled = scaler.transform(X_train_kfold)
        # print(np.amax(X_train_kfold))  # 9490.310668945312
        # print(np.amax(X_train_kfold_scaled))  # 8.236592246485245
        X_test_kfold_scaled = scaler.transform(X_test_kfold)
        #  X_test_scaled = scaler.transform(X_test)  #  this is done in an other place. keeping for legacy

        # pre-trained model
        model = clfGlobals.model
        # ASSUMPTION: sequential model is already trained
        model_name = clfGlobals.get_model_name()

        if model_name != "Sequential":
            model.fit(X_train_kfold_scaled, y_train_kfold)
            y_fold_predicted = model.predict(X_test_kfold_scaled)

        else:
            #  assert (screamGlobals.isTrained is True)
            if clfGlobals.isTrained is True:
                y_fold_predicted = model.predict_classes(X_test_kfold_scaled)
            else:
                print("untrained model is being trained")
                #  create and train default model
                # TODO take to outer function?
                model = create_default_sequential_for_mfcc_size()

                # train model on training set of Kfold
                history = model.fit(X_train_kfold_scaled,
                                    y_train_kfold,
                                    epochs=20,
                                    batch_size=128)

                # lev: to save for this loop only the trained model- uncomment next 2 lines -talk to lev b4 uncommenting
                # save_model_and_weights(model)
                # exit()

                clfGlobals.model = model
                clfGlobals.isTrained = True

                y_fold_predicted = model.predict_classes(X_test_kfold_scaled)

        # print("finished_predictions")
        #  calculate scores

        #  calculate global precision score
        global_score = sklearn.metrics.precision_score(y_test_kfold, y_fold_predicted, average='micro')
        scores_accuracy_k_fold.append(global_score)
        #  take f1_score only for the classifiers class
        f1_score = sklearn.metrics.f1_score(y_test_kfold, y_fold_predicted, pos_label=IntPositive, average='binary')
        scores_f1_k_fold.append(f1_score)
        # recall for positive class
        recall_score = sklearn.metrics.recall_score(y_test_kfold, y_fold_predicted, pos_label=IntPositive,
                                                    average='binary')
        scores_recall_fold.append(recall_score)

    mean_scores_accuracy_k_fold = np.mean(np.array(scores_accuracy_k_fold))
    mean_scores_f1_k_fold = np.mean(np.array(scores_f1_k_fold))
    mean_scores_recall_fold = np.mean(np.array(scores_recall_fold))
    #  print(mean_scores_accuracy_k_fold,mean_scores_f1_k_fold)
    return mean_scores_accuracy_k_fold, mean_scores_f1_k_fold, mean_scores_recall_fold


def get_Repeated_strtfy_results(combined_lower_amount, k, repetitions):
    # prepare dataFrame for stratified and for split
    data_no_fileName = combined_lower_amount.drop(['filename'], axis=1)
    #  print(data_no_fileName.shape)  # ( , ) with label column

    # encode strings of labels to integers
    labels_list = data_no_fileName.iloc[:, -1]
    # print(labels_list.shape)  # (734,)

    clf_labels_amount = len(data_no_fileName.loc[data_no_fileName['label'] == clfGlobals.clf_label])
    nearMiss_clf_amount = len(data_no_fileName.loc[data_no_fileName['label'] == clfGlobals.nearMissLabel])

    encoder = LabelEncoder()
    encoded_labels_csv = np.array(encoder.fit_transform(labels_list))
    #  print(encoded_labels_csv)  # [0 0 0 ... 1 1 1]

    # lev solve bug: should have only 3 labels at this point: <clf_label true>, NearMiss_<clf_name>, and everything else
    clf_label_int = encoder.transform([clfGlobals.clf_label])[0]  # 0 because it returns a list

    # lev fix for 0 amount Near_Miss
    if nearMiss_clf_amount > 0:
        clf_NearMiss_label_int = encoder.transform([clfGlobals.nearMissLabel])[0]  # TODO?
    else:
        clf_NearMiss_label_int = -1  #  -1 is illegal coding
    in_case_int_is_zero = 8  # power of 2 - easier for micro commands
    others_int = clf_label_int + clf_NearMiss_label_int + in_case_int_is_zero  # label for all others

    """
    Old code b4 fix for 0 Near_miss- delete after october 2019
    clf_NearMiss_label_int = encoder.transform([clfGlobals.nearMissLabel])[0]
    in_case_int_is_zero = 2
    others_int = clf_label_int + clf_NearMiss_label_int + in_case_int_is_zero # label for all others
    """
    encoded_labels_csv[(encoded_labels_csv != clf_label_int) & (encoded_labels_csv != clf_NearMiss_label_int)] = others_int
    label_coded_amount = len(encoded_labels_csv[encoded_labels_csv == clf_label_int])
    nearMiss_coded_amount = len(encoded_labels_csv[encoded_labels_csv == clf_NearMiss_label_int])
    assert(label_coded_amount == clf_labels_amount)
    assert (nearMiss_coded_amount == nearMiss_clf_amount)

    # print(encoded_labels_csv.shape) # (734,)

    # print(list(encoder.inverse_transform([0,1])))  #  ['negative', 'scream']

    # take all except labels column. important to add the dtype=float, otherwise im getting an error in the kfold.
    only_features = np.array(data_no_fileName.iloc[:, :-1], dtype=float)
    #  print(only_features.shape)  # ( , )

    """"
    splitting stages
    """
    # split for test and k-fold
    X_for_k_fold, X_test, y_for_k_fold, y_test = train_test_split \
        (only_features, encoded_labels_csv, test_size=clfGlobals.Kfold_testSize, stratify=encoded_labels_csv)
    # print(len(y_for_k_fold))  # 587
    # print(len(y_test))  # 147

    # after stratify- lets keep only binary classification (merging Near Miss and Far miss into one label
    print("merging all negative labels into one label")

    # find the positive label transformation
    clf_label_int = encoder.transform([clfGlobals.clf_label])[0]  # 0 because it returns a list
    #  print(clf_label_int)  # 2

    # give temp values in order to prevent conflicts
    tempIntLabel: int = 111
    IntNegative: int = 0
    IntPositive: int = 1
    y_for_k_fold[y_for_k_fold == clf_label_int] = tempIntLabel
    y_test[y_test == clf_label_int] = tempIntLabel

    y_for_k_fold[y_for_k_fold != tempIntLabel] = IntNegative
    y_test[y_test != tempIntLabel] = IntNegative

    # switch to binary
    y_for_k_fold[y_for_k_fold == tempIntLabel] = IntPositive
    y_test[y_test == tempIntLabel] = IntPositive

    #  now, y_test is binary, 0 represents the true label of the classifier

    """
    # find models best hyper-parameters
    optimised_model= get_optimised_model(X_for_k_fold, X_test, y_for_k_fold, y_test)
    screamGlobals.model= optimised_model
    print(f"finished optimizing our model")
    """

    # important values for deciding which is the best classifier
    scores_accuracy_k_fold_repeated = []
    scores_f1_k_fold_repeated = []
    scores_k_fold_recall = []

    for repeat_number in tqdm(range(1, repetitions)):
        score_mean_k_fold, f1_mean_k_fold, recall_k_fold = get_stratified_results(k, X_for_k_fold,
                                                                                  y_for_k_fold, IntPositive,
                                                                                  clfGlobals)

        scores_accuracy_k_fold_repeated.append(score_mean_k_fold)
        scores_f1_k_fold_repeated.append(f1_mean_k_fold)
        scores_k_fold_recall.append(recall_k_fold)

    mean_accuracy_k_fold_repeated = np.mean(np.array(scores_accuracy_k_fold_repeated))
    mean_f1_k_fold_repeated = np.mean(np.array(scores_f1_k_fold_repeated))
    mean_scores_k_fold_recall = np.mean(np.array(scores_k_fold_recall))

    return mean_accuracy_k_fold_repeated, mean_f1_k_fold_repeated, mean_scores_k_fold_recall


def get_model_head():
    if clfGlobals.model is None:
        raise Exception("No model, supply a model please.")

    model_name = clfGlobals.get_model_name()
    #  additional info will be in a single cell in the CSV so we seperate info by a double-underline:  __
    if model_name == "Sequential":
        additional_info = f'model_layers={len(clfGlobals.model.layers)}'
    else:
        additional_info = 'None'

    csv_model_head = f'model_name model_info'
    csv_model_results = f'{model_name} {additional_info}'

    return csv_model_head, csv_model_results


def results_to_csv(csv_results_head, csv_results):
    # printing results to csv for tracking
    print("results_to_csv")
    csv_data_results = f'{clfGlobals.clf_label} {clfGlobals.get_total_samples()}' \
                       f' {clfGlobals.positives} {clfGlobals.negatives}' \
                       f' {clfGlobals.getInputDim()} {clfGlobals.Kfold_testSize} {clfGlobals.k_folds}' \
                       f' {clfGlobals.k_fold_repetitions} {clfGlobals.sampling_data_repetitions}'

    csv_features_results = f'{clfGlobals.n_mfcc}'

    # TODO- get_model_head() can be optimized in future versions to separate head from results
    csv_model_head, csv_model_results = get_model_head()

    csv_final_results = csv_results + ' ' + csv_data_results + ' ' \
                        + csv_features_results + ' ' + csv_model_results + ' ' + clfGlobals.userInput
    # TODO enter also model params according to moris suggestion- name, hyper params

    if not os.path.exists(clfGlobals.resultsPath):
        csv_data_head = f'clf_label total_samples positives negatives total_features test' \
                        f'_size_ratio folds kfold_repeats' \
                        f' sampling_repeats'

        csv_features_head = f'n_mfcc_amount'

        csv_final_head = csv_results_head + ' ' + csv_data_head + ' ' \
                         + csv_features_head + ' ' + csv_model_head + ' ' + 'userInput'

        # save to csv (append new lines)

        #  file doesnt exist- lets create with header
        file = open(clfGlobals.resultsPath, 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(csv_final_head.split())
    else:
        # file exists with header- just add lines
        file = open(clfGlobals.resultsPath, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(csv_final_results.split())


def experiment_data_size():
    print("executing screamClf flow")
    # change global definitions at the top of the file inside global_For_Clf class #todo lev maybe create config file
    # screamGlobals = global_For_Clf()
    create_csv()  # need to create only once, but for now its already created

    # lev testing parameter
    # screamGlobals.try_lower_amount= lower_bound_per_class  #  this is used in an outer function in a different file

    accuracy_sampling = []  # each iteration will append avg accuracy value
    f1_sampling = []
    recall_sampling = []
    for sample_number in tqdm(range(1, clfGlobals.sampling_data_repetitions)):
        lower_bound_data_panda = create_lower_bound_data_panda(clfGlobals.data_file_path,
                                                               clfGlobals.clf_label)

        score_mean_k_fold, f1_mean_k_fold, recall_k_fold = get_Repeated_strtfy_results \
            (lower_bound_data_panda, clfGlobals.k_folds, clfGlobals.k_fold_repetitions)

        accuracy_sampling.append(score_mean_k_fold)
        f1_sampling.append(f1_mean_k_fold)
        recall_sampling.append(recall_k_fold)

    mean_accuracy_sampling = np.mean(np.array(accuracy_sampling))
    mean_f1_sampling = np.mean(np.array(f1_sampling))
    mean_recall_sampling = np.mean(np.array(recall_sampling))

    # gather results
    csv_results_head = f'total_accuracy f1_{clfGlobals.clf_label} recall_{clfGlobals.clf_label}'

    csv_results = f'{mean_accuracy_sampling} {mean_f1_sampling} {mean_recall_sampling}'

    results_to_csv(csv_results_head, csv_results)


def different_model():
    """
    goal: run with different sample sizes to understand on different models (changed manually inside the script
        the models) and verified with a csv results file which direction should we take ...

        in this function we run also an automated imported algorithm to find best model parameters and then ran
        our algo with our custom kfold on our custom data with spesific ratio's positive:negative and
         negative: nearMiss_negative to find best classifier
    """
    # lower size must be > Nearmiss

    for size in tqdm([100, 150, 200, 250, 300, 350, 400]):
        clfGlobals.try_lower_amount = size
        experiment_data_size()


def get_best_model_results(optimised_model, X_test, y_test, IntPositive, X_for_k_fold, y_for_k_fold):
    y_fold_predicted = optimised_model.predict_classes(X_test)

    print(y_fold_predicted, end='')
    print(y_test, end='')

    accuracy = sklearn.metrics.precision_score(y_test, y_fold_predicted, average='micro')
    #  take f1_score only for the classifiers class
    f1 = sklearn.metrics.f1_score(y_test, y_fold_predicted, pos_label=IntPositive, average='binary')
    # recall for positive class
    recall = sklearn.metrics.recall_score(y_test, y_fold_predicted, pos_label=IntPositive,
                                          average='binary')

    print(f' accuracy, f1, recall: ', accuracy, f1, recall)

    print("now on trained (off the record): ")
    y_fold_predicted = optimised_model.predict_classes(X_for_k_fold)
    print(y_fold_predicted, end='')
    print(y_for_k_fold, end='')

    accuracy_trained = sklearn.metrics.precision_score(y_for_k_fold, y_fold_predicted, average='micro')
    #  take f1_score only for the classifiers class
    f1_trained = sklearn.metrics.f1_score(y_for_k_fold, y_fold_predicted, pos_label=IntPositive, average='binary')
    # recall for positive class
    recall_trained = sklearn.metrics.recall_score(y_for_k_fold, y_fold_predicted, pos_label=IntPositive,
                                                  average='binary')
    print(f' accuracy, f1, recall: ', accuracy_trained, f1_trained, recall_trained)

    return accuracy, f1, recall


def get_best_model(combined_lower_amount):
    """
    first prepares data for test and train groups, then
    returns best model with best hyper-parameters
    """

    # prepare dataFrame for stratified and for split
    data_no_fileName = combined_lower_amount.drop(['filename'], axis=1)
    #  print(data_no_fileName.shape)  # ( , ) with label column

    # encode strings of labels to integers
    labels_list = data_no_fileName.iloc[:, -1]
    # print(labels_list.shape)  # (734,)

    clf_labels_amount = len(data_no_fileName.loc[data_no_fileName['label'] == clfGlobals.clf_label])
    nearMiss_clf_amount = len(data_no_fileName.loc[data_no_fileName['label'] == clfGlobals.nearMissLabel])

    encoder = LabelEncoder()
    encoded_labels_csv = np.array(encoder.fit_transform(labels_list))
    #  print(encoded_labels_csv)  # [0 0 0 ... 1 1 1]

    # lev solve bug: should have only 3 labels at this point: <clf_label true>, NearMiss_<clf_name>, and everything else
    clf_label_int = encoder.transform([clfGlobals.clf_label])[0]  # 0 because it returns a list
    if nearMiss_clf_amount > 0:
        clf_NearMiss_label_int = encoder.transform([clfGlobals.nearMissLabel])[0]  # TODO?
    else:
        clf_NearMiss_label_int = -1  #  -1 is illegal coding
    in_case_int_is_zero = 8  # power of 2 - easier for micro commands
    others_int = clf_label_int + clf_NearMiss_label_int + in_case_int_is_zero  # label for all others

    encoded_labels_csv[(encoded_labels_csv != clf_label_int) & (encoded_labels_csv != clf_NearMiss_label_int)] = others_int
    label_coded_amount = len(encoded_labels_csv[encoded_labels_csv == clf_label_int])
    nearMiss_coded_amount = len(encoded_labels_csv[encoded_labels_csv == clf_NearMiss_label_int])
    assert(label_coded_amount == clf_labels_amount)
    assert (nearMiss_coded_amount == nearMiss_clf_amount)

    # print(encoded_labels_csv.shape) # (734,)

    # print(list(encoder.inverse_transform([0,1])))  #  ['negative', 'scream']

    # take all except labels column. important to add the dtype=float, otherwise im getting an error in the kfold.
    only_features = np.array(data_no_fileName.iloc[:, :-1], dtype=float)
    #  print(only_features.shape)  # ( , )


    """"
    splitting stages
    """
    # split for test and k-fold
    X_for_k_fold, X_test, y_for_k_fold, y_test = train_test_split \
        (only_features, encoded_labels_csv, test_size=clfGlobals.Kfold_testSize, stratify=encoded_labels_csv)
    # print(len(y_for_k_fold))  # 587
    # print(len(y_test))  # 147

    # after stratify- lets keep only binary classification (merging Near Miss and Far miss into one label
    print("merging all negative labels into one label")

    # find the positive label transformation
    clf_label_int = encoder.transform([clfGlobals.clf_label])[0]  # 0 because it returns a list
    #  print(clf_label_int)  # 2

    # give temp values in order to prevent conflicts
    tempIntLabel: int = 111
    IntNegative: int = 0
    IntPositive: int = 1
    y_for_k_fold[y_for_k_fold == clf_label_int] = tempIntLabel
    y_test[y_test == clf_label_int] = tempIntLabel

    y_for_k_fold[y_for_k_fold != tempIntLabel] = IntNegative
    y_test[y_test != tempIntLabel] = IntNegative

    # switch to binary
    y_for_k_fold[y_for_k_fold == tempIntLabel] = IntPositive
    y_test[y_test == tempIntLabel] = IntPositive

    #  now, y_test is binary, 0 represents the true label of the classifier

    # find models best hyper-parameters
    optimised_model = get_optimised_model_final(X_for_k_fold, X_test, y_for_k_fold, y_test)
    clfGlobals.model = optimised_model
    print(f"finished optimizing our model")

    accuracy, f1, recall = get_best_model_results(optimised_model,
                                                  X_test, y_test, IntPositive, X_for_k_fold, y_for_k_fold)

    return optimised_model, accuracy, f1, recall


def model_results_to_csv(csv_results_head, csv_results):
    # printing results to csv for tracking
    print("results_to_csv")
    csv_data_results = f'{clfGlobals.clf_label} {clfGlobals.get_total_samples()}' \
                       f' {clfGlobals.positives} {clfGlobals.negatives}' \
                       f' {clfGlobals.getInputDim()} {clfGlobals.Kfold_testSize} {clfGlobals.k_folds}' \
                       f' {clfGlobals.k_fold_repetitions} {clfGlobals.sampling_data_repetitions}'

    csv_features_results = f'{clfGlobals.n_mfcc}'

    # TODO- get_model_head() can be optimized in future versions to separate head from results
    csv_model_head, csv_model_results = get_model_head()

    csv_final_results = csv_results + ' ' + csv_data_results + ' ' + csv_features_results + ' ' + csv_model_results
    # TODO enter also model params according to moris suggestion- name, hyper params

    if not os.path.exists(clfGlobals.resultsPath):
        csv_data_head = f'clf_label total_samples positives negatives total_features test' \
                        f'_size_ratio folds kfold_repeats' \
                        f' sampling_repeats'

        csv_features_head = f'n_mfcc_amount'

        csv_final_head = csv_results_head + ' ' + csv_data_head + ' ' + csv_features_head + ' ' + csv_model_head

        # save to csv (append new lines)

        #  file doesnt exist- lets create with header
        file = open(clfGlobals.resultsPath, 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(csv_final_head.split())
    else:
        # file exists with header- just add lines
        file = open(clfGlobals.resultsPath, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(csv_final_results.split())


def save_model_and_weights(best_model):
    """
    :param best_model:

        new keras version allows saving a trained-model with weights in one line of code
    """
    model_path = str(f'{clfGlobals.finalModelsPath}/{clfGlobals.clf_label}_clf_mfcc_{clfGlobals.n_mfcc}.h5')
    print(model_path)

    # new version doesnt require weights
    best_model.save(model_path)
    print("Saved model to disk")


def save_best_model():
    """
    goal: after getting results with <different_model()> , we now have a much more limited search range in which
    we will find our best model
    """
    print("executing screamClf flow- save best model")

    # change global definitions at the top of the file inside global_For_Clf class #todo lev maybe create config file
    # screamGlobals = global_For_Clf()
    create_csv()  # need to create only once, but for now its already created

    # lev testing parameter
    # screamGlobals.try_lower_amount= lower_bound_per_class  #  this is used in an outer function in a different file

    lower_bound_data_panda = create_lower_bound_data_panda(clfGlobals.data_file_path,
                                                           clfGlobals.clf_label)

    optimised_model, accuracy, f1, recall = get_best_model(lower_bound_data_panda)

    # gather results
    csv_results_head = f'total_accuracy f1_{clfGlobals.clf_label} recall_{clfGlobals.clf_label}'

    # csv_results = f'{mean_accuracy_sampling} {mean_f1_sampling} {mean_recall_sampling}'
    csv_results = f'{accuracy} {f1} {recall}'

    model_results_to_csv(csv_results_head, csv_results)

    save_model_and_weights(optimised_model)


def load_bestModel():
    # load model
    """
    assumption: the best model is already saved using the save_best_model() function
    if assumption is incorrect it will run save_best_model()
    """

    model_path = str(f'{clfGlobals.finalModelsPath}/{clfGlobals.clf_label}_clf_mfcc_{clfGlobals.n_mfcc}.h5')
    if not os.path.exists(model_path):
        print(f'{model_path} doesnt exist, running save_best_model() to save best model')
        save_best_model()
        print("finished saving model- continue with loading")

    print(f"loading model from {model_path}")
    model = load_model(model_path)
    model.summary()
    clfGlobals.isTrained = True
    return model


def experiment_data_size_with_model():
    print("executing clfGlobals flow")
    # change global definitions at the top of the file inside global_For_Clf class #todo lev maybe create config file
    # screamGlobals = global_For_Clf()
    create_csv()  # need to create only once, but for now its already created

    # lev testing parameter
    # screamGlobals.try_lower_amount= lower_bound_per_class  #  this is used in an outer function in a different file

    accuracy_sampling = []  # each iteration will append avg accuracy value
    f1_sampling = []
    recall_sampling = []
    for sample_number in tqdm(range(1, clfGlobals.sampling_data_repetitions)):
        lower_bound_data_panda = create_lower_bound_data_panda(clfGlobals.data_file_path,
                                                               clfGlobals.clf_label)

        score_mean_k_fold, f1_mean_k_fold, recall_k_fold = get_Repeated_strtfy_results \
            (lower_bound_data_panda, clfGlobals.k_folds, clfGlobals.k_fold_repetitions)

        accuracy_sampling.append(score_mean_k_fold)
        f1_sampling.append(f1_mean_k_fold)
        recall_sampling.append(recall_k_fold)

    mean_accuracy_sampling = np.mean(np.array(accuracy_sampling))
    mean_f1_sampling = np.mean(np.array(f1_sampling))
    mean_recall_sampling = np.mean(np.array(recall_sampling))

    # gather results
    csv_results_head = f'total_accuracy f1_{clfGlobals.clf_label} recall_{clfGlobals.clf_label}'

    csv_results = f'{mean_accuracy_sampling} {mean_f1_sampling} {mean_recall_sampling}'

    if clfGlobals.resultsPath == 'results/'+str(clfGlobals.get_clf_label())+'/experiment3.csv':
        print("CSV for experiment3")
        results_to_csv_experiment3(csv_results_head, csv_results)
    else:
        results_to_csv(csv_results_head, csv_results)


def evaluate_model(model):
    """
    different_model() , but updated with model as input

    goal: run with different sample sizes to understand on different models (changed manually inside the script
        the models) and verified with a csv results file which direction should we take ...

        in this function we run also an automated imported algorithm to find best model parameters and then ran
        our algo with our custom kfold on our custom data with spesific ratio's positive:negative and
         negative: nearMiss_negative to find best classifier
    """
    # lower size must be > Nearmiss
    clfGlobals.model = model  # 100,150,200,250,300,350,400
    #  TODO can automate the process of choosing data sizes
    for size in tqdm([100, 150, 200, 250, 300, 350, 400]):
        clfGlobals.try_lower_amount = size
        experiment_data_size_with_model()


def evaluate_model_different_data_size(model):
    """
    different_model() , but updated with model as input

    goal: run with different sample sizes to understand on different models (changed manually inside the script
        the models) and verified with a csv results file which direction should we take ...

        in this function we run also an automated imported algorithm to find best model parameters and then ran
        our algo with our custom kfold on our custom data with spesific ratio's positive:negative and
         negative: nearMiss_negative to find best classifier
    """
    # lower size must be > Nearmiss
    clfGlobals.model = model
    #  TODO can automate the process of choosing data sizes-
    #  cry - for cry we have much less than scream so i divided by 5 to get same samples amount on X axis as in scream
    for size in tqdm([50,75, 100, 125, 150]):
        print(f"checking size {size}")
        clfGlobals.try_lower_amount = size
        experiment_data_size_with_model()


def evaluate_best_model():
    model = load_bestModel()
    evaluate_model(model)


def save_evaluate_bestModel():
    save_best_model()
    evaluate_best_model()


def compare_different_models():
    """
    save to csv results from different models
    """
    # create all the relevant models
    # load best_model with best hyper parameters- this one is already trained
    clfGlobals.userInput = 'compare_different_models-10%_nearmiss_instead_of_50%'

    best_model = load_bestModel()
    # create other models TODO - can put in different function all the creations of models
    id3 = Id3Estimator()
    knn = KNeighborsClassifier(3)
    svc1 = SVC(kernel="linear", C=0.025)
    svc2 = SVC(gamma=2, C=1)
    gauss = GaussianProcessClassifier(1.0 * RBF(1.0))
    decisionTrees = DecisionTreeClassifier(max_depth=25)
    rand = RandomForestClassifier(max_depth=25, n_estimators=10, max_features=1)
    adaboost = AdaBoostClassifier()
    gaussNB = GaussianNB()
    mlp = MLPClassifier(alpha=1, max_iter=100)

    models_to_check = [best_model, id3, knn, svc1, svc1, svc2, gauss, decisionTrees, rand, gaussNB, adaboost, mlp]

    for model in tqdm(models_to_check):
        name = type(model).__name__
        print(f'checking model: {name}')
        evaluate_model(model)


def get_date():
    """
    :return: current date with in the next format:   2019-08-21
    """

    mylist = []
    today = datetime.date.today()
    mylist.append(today)
    date = mylist[0]
    return date


def get_models_to_check():
    """

    :return: a list of models (classifiers)
    """
    # load best_model with best hyper parameters- this one is already trained
    best_model = load_bestModel()
    id3 = Id3Estimator()
    knn = KNeighborsClassifier(3)
    svc1 = SVC(kernel="linear", C=0.025)
    svc2 = SVC(gamma=2, C=1)
    gauss = GaussianProcessClassifier(1.0 * RBF(1.0))
    decisionTrees = DecisionTreeClassifier(max_depth=25)
    rand = RandomForestClassifier(max_depth=25, n_estimators=10, max_features=1)
    adaboost = AdaBoostClassifier()
    gaussNB = GaussianNB()
    mlp = MLPClassifier(alpha=1, max_iter=100)

    models_to_check = [best_model, id3, knn, svc1, svc2, gauss, decisionTrees, rand, gaussNB, adaboost, mlp]
    return models_to_check


def experiment1():
    """
    save to csv results from different models with different dataset sizes

    check how different data size influences the accuracy and whether the accuracy converges/decreases.
    """

    # set global variables for experiment
    date = get_date()
    clfGlobals.userInput = f'experiment_1_date_{date}'
    clfGlobals.data_file_path = 'csv/'+str(clfGlobals.get_clf_label())+'/data_experiment_1.csv'

    # get all the relevant models
    models_to_check = get_models_to_check()

    for model in tqdm(models_to_check):
        name = type(model).__name__
        print(f'checking model: {name}')
        evaluate_model_different_data_size(model)


def create_csv_different_mfcc():
    """
    input: uses screamGlobals for input
    output: .csv file with screamGlobals.csv_initial_head columns
    """
    # important variables
    data_file_path = clfGlobals.data_file_path
    min_wav_duration = clfGlobals.min_wav_duration
    #  print(data_file_path, min_wav_duration)
    """
    #  prevent data file over run by accident
    if os.path.exists(data_file_path):
        text = input(f'Press the space bar to override {data_file_path} and continue with the script')
        if text != ' ':
            sys.exit('User aborted script, data file saved :)')
    """

    #  covering- allow running over the csv for faster results for experiment 2

    if os.path.exists(data_file_path):
        # verify table fits the mfcc number- if True- return (continue with script as usuall), else- raise Error
        n_mfcc_number = clfGlobals.n_mfcc
        with open(data_file_path) as csvFile:
            reader = csv.reader(csvFile)
            field_names_list = next(reader)  # read first row only (header)
            mfcc_list = [x for x in field_names_list if x.startswith("mfcc")]
            len_actual_mfcc_features = len(mfcc_list)
        if len_actual_mfcc_features == n_mfcc_number:
            print(f'OK: {len_actual_mfcc_features} ==  n_mfcc_number={n_mfcc_number}')
            return
        else:
            raise Exception(f'len_actual_mfcc_features'
                            f'(mfcc inside {data_file_path}={len_actual_mfcc_features},'
                            f' but n_mfcc_number(inside globals class of this script)={n_mfcc_number},'
                            f' values must be equal.')

    # create header for csv
    header = clfGlobals.csv_initial_head
    fcc_amount = clfGlobals.n_mfcc
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


def experiment_data_size_with_model_different_mfcc():
    print("executing screamClf flow")
    # change global definitions at the top of the file inside global_For_Clf class #todo lev maybe create config file
    # screamGlobals = global_For_Clf()
    create_csv_different_mfcc()  # need to create only once, but for now its already created

    # lev testing parameter
    # screamGlobals.try_lower_amount= lower_bound_per_class  #  this is used in an outer function in a different file

    accuracy_sampling = []  # each iteration will append avg accuracy value
    f1_sampling = []
    recall_sampling = []
    for sample_number in tqdm(range(1, clfGlobals.sampling_data_repetitions)):
        lower_bound_data_panda = create_lower_bound_data_panda(clfGlobals.data_file_path,
                                                               clfGlobals.clf_label)

        score_mean_k_fold, f1_mean_k_fold, recall_k_fold = get_Repeated_strtfy_results \
            (lower_bound_data_panda, clfGlobals.k_folds, clfGlobals.k_fold_repetitions)

        accuracy_sampling.append(score_mean_k_fold)
        f1_sampling.append(f1_mean_k_fold)
        recall_sampling.append(recall_k_fold)

    mean_accuracy_sampling = np.mean(np.array(accuracy_sampling))
    mean_f1_sampling = np.mean(np.array(f1_sampling))
    mean_recall_sampling = np.mean(np.array(recall_sampling))

    # gather results
    csv_results_head = f'total_accuracy f1_{clfGlobals.clf_label} recall_{clfGlobals.clf_label}'

    csv_results = f'{mean_accuracy_sampling} {mean_f1_sampling} {mean_recall_sampling}'

    results_to_csv(csv_results_head, csv_results)


def experiment2():
    """
    save to csv results from different models with different mfcc sizes
    """
    print("experiment2 in progress")
    # set global variables for experiment
    date = get_date()
    clfGlobals.userInput = f'experiment_2_date_{date}'

    # get all the relevant models
    models_to_check = get_models_to_check()

    #  logically the loops should be switched, but it's better for mfcc loop to be the outer one for faster run-time
    # 1, 5, 10, 12, 15, 20, 25, 30, 35, 40
    for size in tqdm([1, 5, 10, 12, 15, 20, 25, 30, 35, 40]):
        clfGlobals.n_mfcc = size

        for model in tqdm(models_to_check):
            name = type(model).__name__
            print(f'checking model: {name}')
            clfGlobals.model = model

            clfGlobals.isTrained = False  # False will make the algorithm build and train sequential_<MFCC_amount>

            mfcc_amount = size
            clfGlobals.data_file_path = f'csv/'+str(clfGlobals.get_clf_label())+f'/data_experiment_2_mfcc_{mfcc_amount}.csv'

            experiment_data_size_with_model_different_mfcc()


def experiments():
    # different_model()
    # save_evaluate_bestModel()
    # load_best_run_on_test()
    compare_different_models()


def results_to_csv_experiment3(csv_results_head, csv_results):
    """
    Created this func especially for experiment3.
    the difference from original func "results_to_csv" is that here we also calculate
    near-miss elements into the CSV and i didnt want to mess with the CSV format of previous/future results.
    """

    # printing results to csv for tracking
    print("results_to_csv for experiment3")
    csv_data_results = f'{clfGlobals.clf_label} {clfGlobals.get_total_samples()}' \
                       f' {clfGlobals.positives} {clfGlobals.negatives}' \
                       f' {clfGlobals.nearMissRatio} {clfGlobals.nearMiss_samples}' \
                       f' {clfGlobals.getInputDim()} {clfGlobals.Kfold_testSize} {clfGlobals.k_folds}' \
                       f' {clfGlobals.k_fold_repetitions} {clfGlobals.sampling_data_repetitions}'

    csv_features_results = f'{clfGlobals.n_mfcc}'

    # TODO- get_model_head() can be optimized in future versions to separate head from results
    csv_model_head, csv_model_results = get_model_head()

    csv_final_results = csv_results + ' ' + csv_data_results + ' ' \
                        + csv_features_results + ' ' + csv_model_results + ' ' + clfGlobals.userInput
    # TODO enter also model params according to moris suggestion- name, hyper params

    if not os.path.exists(clfGlobals.resultsPath):
        csv_data_head = f'clf_label total_samples positives negatives nearMiss_ratio nearMiss_samples' \
                        f' total_features test_size_ratio folds kfold_repeats' \
                        f' sampling_repeats'

        csv_features_head = f'n_mfcc_amount'

        csv_final_head = csv_results_head + ' ' + csv_data_head + ' ' \
                         + csv_features_head + ' ' + csv_model_head + ' ' + 'userInput'

        # save to csv (append new lines)

        #  file doesnt exist- lets create with header
        file = open(clfGlobals.resultsPath, 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(csv_final_head.split())
    else:
        # file exists with header- just add lines
        file = open(clfGlobals.resultsPath, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(csv_final_results.split())


def experiment3():
    """
    save to csv results from different models with different NearMiss ratio
    """
    print("experiment3 in progress")
    # set global variables for experiment
    date = get_date()
    clfGlobals.userInput = f'experiment_3_date_{date}'
    clfGlobals.resultsPath = 'results/'+str(clfGlobals.get_clf_label())+'/experiment3.csv'

    # get all the relevant models
    models_to_check = get_models_to_check()
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    # cry- changing amounts to prevent less than 2 near misses
    # TODO because we dont have a lot of positives- its hard to get good results- first place to enhance later
    # TODO  when we will have more Positive samples
    nearMiss_ratios = [1, 2, 4, 6, 8]
    for ratio in tqdm(nearMiss_ratios):
        clfGlobals.nearMissRatio = ratio

        for model in tqdm(models_to_check):
            name = type(model).__name__
            print(f'checking model: {name}')
            clfGlobals.model = model

            experiment_data_size_with_model()


def experiment5():
    """
    save to csv results from different models
    """
    print("experiment5 in progress")
    # set global variables for experiment
    date = get_date()
    clfGlobals.userInput = f'experiment_5_date_{date}'
    clfGlobals.n_mfcc = 20  # verifying correct value for experiment

    # get all the relevant models
    models_to_check = get_models_to_check()

    for model in tqdm(models_to_check):
        name = type(model).__name__
        print(f'checking model: {name}')
        clfGlobals.model = model

        # assumption: experiment 2 was already done at this point. use experiment 2 csv.
        clfGlobals.data_file_path = f'csv/'+str(clfGlobals.get_clf_label())+'/data_experiment_2_mfcc_20.csv'

        experiment_data_size_with_model_different_mfcc()

    # now check best model with 15 as mfcc resolution
    clfGlobals.n_mfcc = 15
    clfGlobals.data_file_path = f'csv/'+str(clfGlobals.get_clf_label())+'/data_experiment_2_mfcc_15.csv'

    clfGlobals.model = load_bestModel()
    experiment_data_size_with_model()


def experiments_for_report():
    """
    perform experiments for the final report
    """
    #experiment1()
    experiment2()
    #experiment3()
    #experiment5()


if __name__ == "__main__":
    # todo add user input- basic clf for mori-use result
    # clfGlobals = global_For_Clf('cry')  # create global variable
    #clfGlobals = global_For_Clf('scream')  # create global variable
    #experiments_for_report()
    # 'gasp','whisper' ,'sniff'

    for clf_name in ['sniff']:
        print(f'checking {clf_name} classification')
        clfGlobals = global_For_Clf(clf_name)  # create global variable
        experiments_for_report()

