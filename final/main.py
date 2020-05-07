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


# loading data
def load_data():
    audio_path = 'train/positive/scream/1_scream_female_room.wav'
    data, sampling_rate = librosa.load(audio_path)

    # print(sampling_rate)
    # print(data)

    return data, sampling_rate


# waveplot
def save_waveplot(data, sampling_rate):
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(data, sr=sampling_rate)
    title = "Scream waveplot"
    plt.title(title)
    plt.savefig('plots/drafts/' + title, bbox_inches='tight')

    # Zooming in
    n0 = 9000
    n1 = 9100
    plt.figure(figsize=(14, 5))
    plt.plot(data[n0:n1])
    plt.grid()
    plt.savefig('plots/drafts/' + title + '_zero_crossing', bbox_inches='tight')

    # verifying amount of zero crosses
    zero_crossings = librosa.zero_crossings(data[n0:n1], pad=False)
    print('zero crossings: ' + str(sum(zero_crossings)))

    print('spectral centroid samples amount: ')
    spectral_centroids = librosa.feature.spectral_centroid(data, sr=sampling_rate)[0]
    print(spectral_centroids.shape)

    # Computing the time variable for visualization
    frames = range(len(spectral_centroids))
    time = librosa.frames_to_time(frames)

    # Plotting the Spectral Centroid along the waveform
    librosa.display.waveplot(data, sr=sampling_rate, alpha=0.4)
    plt.plot(time, normalize(spectral_centroids), color='r')
    plt.savefig('plots/drafts/' + title + '_spectral_centroids', bbox_inches='tight')

    # spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(data + 0.01, sr=sampling_rate)[0]
    librosa.display.waveplot(data, sr=sampling_rate, alpha=0.4)
    plt.plot(time, normalize(spectral_rolloff), color='g')
    plt.savefig('plots/drafts/' + title + '_spectral_rolloff', bbox_inches='tight')


# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# mfcc feature (normalized)
def save_specshow(data, sampling_rate):
    n_mfcc = 40  # 12 #40
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc)

    # print(mfccs.shape)
    # print(type(mfccs))

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    title = "Scream specshow mfccs=40"
    plt.title(title)
    plt.colorbar()
    plt.savefig('plots/drafts/' + title, bbox_inches='tight')

    # to the same with scaling
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc)
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    print(mfccs.mean(axis=1))
    print(mfccs.var(axis=1))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    title = "Scream specshow mfccs=40 with scale"
    plt.title(title)
    plt.colorbar()
    plt.savefig('plots/drafts/' + title, bbox_inches='tight')


def save_chroma_stft(data, sampling_rate):
    hop_length = 512
    chromagram = librosa.feature.chroma_stft(data, sr=sampling_rate, hop_length=hop_length)
    plt.figure(figsize=(15, 5))
    title = "Scream chroma_stft normalized"
    plt.title(title)
    librosa.util.normalize(chromagram)
    # librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length)
    plt.savefig('plots/drafts/' + title, bbox_inches='tight')
    print(chromagram[6])


def save_rms(data, sampling_rate):
    n_mfcc = 40  # 12 #40
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc)
    S, phase = librosa.magphase(mfccs)
    rms = librosa.feature.rms(S=S)

    plt.figure(figsize=(15, 5))
    plt.subplot(2, 1, 1)
    title = "Scream rms "
    plt.title(title)
    plt.semilogy(rms.T, label='RMS Energy')
    plt.xticks([])
    plt.xlim([0, rms.shape[-1]])
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')

    plt.savefig('plots/drafts/' + title, bbox_inches='tight')


def play_showing_data(data, sampling_rate):
    n_mfcc = 40
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc)
    print("mfccs printed:")
    print(type(mfccs))
    print(mfccs)
    print("size= " + str(mfccs.size))
    print("shape: " + str(mfccs.shape))

    print("mfcc[0] printed:")
    print(type(mfccs[0]))
    print(mfccs[0])
    print("size= " + str(mfccs[0].size))
    print("shape: " + str(mfccs[0].shape))

    index = 0
    mfcc_means = []
    for frequncyRate in mfccs:
        mfcc_means.append(np.mean(frequncyRate))
        index += 1
        print("loop frequncyRate " + str(index) + " , shape:" + str(frequncyRate.shape) + " e mean= " + str(
            np.mean(frequncyRate)))

        # print("e= " + str(e) + "e mean= "+ str(np.mean(e)) )
    print("means list is of size " + str(len(mfcc_means)))
    print(mfcc_means)
    scaler = sklearn.preprocessing.MinMaxScaler()
    X = scaler.fit((np.array(mfcc_means)).reshape(-1, 1))
    print("scaling tool details:")
    print(X)
    print("normalized data:")
    print(scaler.transform((np.array(mfcc_means)).reshape(-1, 1)))


def soundApi():
    print("soundApi")

    client = freesound.FreesoundClient()
    client.set_token("1TUvgpT0yJzOjjez0U5LrSLyWkwhWyJduZ3JK2YC", "token")

    results = client.text_search(query="scream", filter="type:wav duration:[0 TO 5]", fields="id,name")

    print(results)
    for result in results:
        # print(type(result)) # printes Sound class
        print(result)

        """
        lev:
        the method below is correct, but its not as simple- it appears i need to use
        some oath algorithm...for now i'm putting it aside. documented in word the situation,
          #result.retrieve(".",result.name+".wav")       
        """


def splitMyWaves(path, durationInSec):
    """pip install pydub"""

    durationInSec = int(durationInSec)
    os.chdir(path)
    for fileName in tqdm(glob.glob("*.wav")):
        # print(fileName)
        src = path + fileName
        # print(src)
        edges = librosa.effects.split(audio, top_db=40, frame_length=128, hop_length=32)


def extract_feature_to_csv(wav_path, label, data_file_path, min_wav_duration):
    # creating csv

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
    """
    except Exception as e:
        print(wav_name)
        print("quitting due to error")
        exit()
    """
    """
    #  envelope of a waveform
    # deciding not to use for now- reson: keep model simple

    wav_stft = librosa.stft(wav_data)
    feature_wav_envelope = librosa.amplitude_to_db(abs(wav_stft))

    #  print(librosa.get_duration(wav_data,sampling_rate))
    #  print(feature_wav_envelope.shape) #  (1025, 108)
    #  print(feature_wav_envelope[0].shape)
    #  print(feature_wav_envelope[0][:5])

    """
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
        n_mfcc = 40  # resolution amount
        mfccs = librosa.feature.mfcc(y=wav_data, sr=sampling_rate, n_mfcc=n_mfcc)
        S, phase = librosa.magphase(mfccs)
        rms = librosa.feature.rms(S=S)
        #  print(rms.shape)

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
                    f' {np.mean(rms)}'
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
    # important variables
    data_file_path = 'data.csv'
    min_wav_duration = 0.5  # wont use shorter wav files

    #  prevent data file over run by accident
    if os.path.exists(data_file_path):
        text = input(f'Press the space bar to override {data_file_path} and continue with the script')
        if text != ' ':
            sys.exit('User aborted script, data file saved :)')

    # create header for csv
    header = 'filename spectral_centroid zero_crossings spectral_rolloff chroma_stft rms'  # TODO lev-future_improvement edit/add to get better results
    fcc_amount = 40  # lev's initial value here was 40- this is the feature resolution- usually between 12-40
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
                lev- updating to diffrintiate near misses and far misses.
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
                    extract_feature_to_csv(wav_path, label, data_file_path, min_wav_duration)
                except NoBackendError as e:
                    print("audioread.NoBackendError " + "for wav path " + str(wav_path))
                    continue  # one file didnt work, continue to next one

    # [x for x in p.iterdir() if x.is_dir()]
    # print(labels)
    # for label in labels:
    # our_classes= os.listdir(os.join(""))
    # for our_class in
    # path= "train\\"+label


# load *.wav positive *.wav negative into one instance with labels for each
# extract features
# normalize/scale
# pass to keras

class fileNameExceptions(Exception):
    def __init__(self, foo):
        self.foo = foo


def try_catch():
    name = "lev"
    try:
        raise Exception("a")
    except Exception as e:
        print(name)

    print("always")


def bug_aaaah():
    # important variables
    data_file_path_bug = 'ah_bug.csv'
    min_wav_duration = 0.5  # wont use shorter wav files
    bugfilePath = Path(r'train\positive\scream\aaaah.wav')

    # create header for csv
    header = 'filename spectral_centroid zero_crossings spectral_rolloff chroma_stft rms'  # TODO lev-future_improvement edit/add to get better results
    fcc_amount = 40  # lev's initial value here was 40- this is the feature resolution- usually between 12-40
    for i in range(1, fcc_amount + 1):
        header += f' mfcc_{i}'
    header += ' label'
    header = header.split()  # split by spaces as default

    file = open(data_file_path_bug, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    try:
        # bug file
        extract_feature_to_csv(bugfilePath, 'scream', data_file_path_bug, min_wav_duration)
    except NoBackendError as e:
        print(e)
    # working file
    extract_feature_to_csv(Path(r'train\positive\scream\aaa.wav'), 'scream', data_file_path_bug, min_wav_duration)


def csv_ready(csv_path):
    """"
    took lines from here to different functions
    """
    #  process CSV
    print("process CSV")
    # use Pandas package for reading csv- turns the csv into a dataframe
    data_csv = pd.read_csv(csv_path)
    #  print(data_csv.head())
    #  print(data_csv.shape)
    # Dropping unnecessary columns
    data_no_fileName = data_csv.drop(['filename'], axis=1)

    # encode strings of labels to integers
    labels_list = data_no_fileName.iloc[:, -1]
    #  print(labels_list)
    encoder = LabelEncoder()
    encoded_labels_csv = encoder.fit_transform(labels_list)
    #  print(encoded_labels_csv)  #  [0 0 0 ... 1 1 1]

    # splitting data

    # important to use scaling only after splitting the data into train/validation/test
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data_no_fileName.iloc[:, :-1], dtype=float))  # takes all except labels column

    #  print(type(X))  #  <class 'numpy.ndarray'>
    #  np.savetxt('test.csv', X, delimiter=',')  # X is an array
    #  print(np.amax(X))  #10.041361690907811
    #  print(np.amin(X))  #-9.474126604795927


def how_to_stratified_k_fold(k, pkl_path):
    # load combined dataset as pandas dataframe (50%-50% ratio between labels)
    combined_lower_amount = pd.read_pickle(pkl_path)
    # print(combined_lower_amount.shape)  #(734, 47)
    # prepare dataframe for stratified
    data_no_fileName = combined_lower_amount.drop(['filename'], axis=1)
    # print(data_no_fileName.shape) # (734, 46)

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
    # print(only_features.shape)  # (734, 45)

    # stratified split
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    skf.get_n_splits(only_features, encoded_labels_csv)  # StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    # print(skf)

    for train_index, test_index in skf.split(only_features, encoded_labels_csv):
        X_train, X_test = only_features[train_index], only_features[test_index]
        y_train, y_test = encoded_labels_csv[train_index], encoded_labels_csv[test_index]

        """
        lev- verifying the expected behavior

        # print("TRAIN:", train_index, "TEST:", test_index)
        # print("train_index.shape "+ str(train_index.shape))  # train_index.shape (586,)
        # print("test_index.shape " + str(test_index.shape)) # test_index.shape (148,)
        # print("X_train.shape "+ str(X_train.shape))  # X_train.shape (586, 45)
        # print("y_train.shape " + str(y_train.shape))  # y_train.shape (586,)
        # print("X_test.shape "+ str(X_test.shape))  # X_test.shape (148, 45)
        # print("y_test.shape " + str(y_test.shape))  # y_test.shape (148,)
        # unique, counts = np.unique(y_train, return_counts=True)
        # print(dict(zip(unique, counts)))  # {0: 293, 1: 293}
        # unique, counts = np.unique(y_test, return_counts=True)
        # print(dict(zip(unique, counts)))  # {0: 74, 1: 74}

        """
        normalize_builtClassifier(X_train, y_train, X_test, y_test)
        break  # todo delete the break after checks are done


def prepare_for_stratford(csv_path, label):
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

    # split to negatives and positives - taking random rows
    data_csv_positives = data_csv[data_csv.label == label]
    data_csv_negatives = data_csv[data_csv.label != label]

    # create pandas dataframe with lower_amount rows for each label
    positives_lower_amount_samples = data_csv_positives.sample(n=lower_amount)
    negatives_lower_amount_samples = data_csv_negatives.sample(n=lower_amount)
    # print(len(positives_lower_amount_samples[positives_lower_amount_samples.label == label]))
    # print(len(negatives_lower_amount_samples[negatives_lower_amount_samples.label != label]))
    # print(negatives_lower_amount_samples.filename)  #verified it took random rows

    # TODO lev- need to write code lines to take each time different negatives for same positive (suggestion: delete selected rows and then sample again...add if to verify theres enough neg samples each time)

    # combined
    combined_lower_amount = positives_lower_amount_samples
    # have to assign, returns appended datadrame
    combined_lower_amount = combined_lower_amount.append(negatives_lower_amount_samples)
    # print(len(combined_lower_amount))  # 734 ,when lower bound: 367

    # saving pandas dataframe to pickle because i'll continue tomorrow- just for practise
    combined_lower_amount.to_pickle("pickle/combined_lower_amount.pkl")

    # using K=5 in stratified cross validation because k=5 ==> 20% for testing which corresponds for


def splittingData(k, pkl_path):
    """"
    arrange data for split, then split into test and k-fold
    """

    # load combined dataset as pandas dataframe (50%-50% ratio between labels)
    combined_lower_amount = pd.read_pickle(pkl_path)

    # prepare dataframe for stratified and for split
    data_no_fileName = combined_lower_amount.drop(['filename'], axis=1)
    # print(data_no_fileName.shape) # (734, 46)

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
    # print(only_features.shape)  # (734, 45)

    """"
    splitting stages
    """
    # split for test and k-fold
    X_for_k_fold, X_test, y_for_k_fold, y_test = train_test_split(only_features, encoded_labels_csv, test_size=0.2)
    # print(len(y_for_k_fold))  # 587
    # print(len(y_test))  # 147

    # save test data as pkl
    path_x_test = Path("pickle/X_test.pkl")
    with path_x_test.open('wb') as file:
        pkl.dump(X_test, file)
    path_y_test = Path("pickle/y_test.pkl")
    with path_y_test.open('wb') as file:
        pkl.dump(y_test, file)

    # stratified split
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    skf.get_n_splits(X_for_k_fold, y_for_k_fold)  # StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    # print(skf)

    fold_num = 1  # use for saving pkl files with different names
    path = Path('pickle/folds')
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

        # normalize_builtClassifier(X_train_kfold, y_train_kfold, X_test_kfold, y_test_kfold)


def normalize_builtClassifier(k):
    """"
    important to use scaling only after splitting the data into train/validation/test
    scale on training set only, then use the returend "fit" parameters to scale validation and test
    """
    # load from pickle test data
    path_x_test = Path("pickle/X_test.pkl")
    with path_x_test.open('rb') as file:
        X_test_loaded = pkl.load(file)
    path_y_test = Path("pickle/y_test.pkl")
    with path_y_test.open('rb') as file:
        y_test_loaded = pkl.load(file)

    # print(X_test_loaded.shape)  # (147, 45)
    # print(y_test_loaded.shape)  # (147,)

    # built loop from 1-k including upper bound...first of all load pickle, then normalize, then send to keras...
    path = Path('pickle/folds')
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
        #here mori will give me a pre-trained model

        model = models.Sequential()

        # print(X_train_kfold_scaled.shape[1])  #  45 (is the number of columns for each sample)
        model.add(layers.Dense(256, activation='relu', input_shape=(X_train_kfold_scaled.shape[1],)))

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
    #  main:
    # create_csv()
    # csv_ready('data.csv')
    # prepare_for_stratford('data.csv', 'scream')
    # how_to_stratified_k_fold(5,'pickle/combined_lower_amount.pkl')
    # splittingData(5,'pickle/combined_lower_amount.pkl')
    normalize_builtClassifier(5)

# data, sampling_rate= load_data()
#  save_waveplot(data,sampling_rate)
#  save_specshow(data,sampling_rate)
# save_chroma_stft(data,sampling_rate)
# save_rms(data,sampling_rate)
# splitMyWaves("C:\\Users\\tunik\\PycharmProjects\\AI_distressRecognition\\train\\negative\\scream\\", 5)
#  play_showing_data(data, sampling_rate)
# soundApi()
#    try_catch()
# bug_aaaah()

