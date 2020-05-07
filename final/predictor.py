#  imports
import sys
import librosa
import librosa.display
from os import listdir
from os.path import isfile, join
import csv
import sys
from keras import models
from keras import layers
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
from shutil import copyfile

import shutil

import datetime
#
#the libary for the window dialog to select file
import tkinter as tk
from tkinter import filedialog
#
#
#
#
# Import the AudioSegment class for processing audio and the
# split_on_silence function for separating out silent chunks.
from pydub import AudioSegment
from pydub.silence import split_on_silence
import pydub
# from evaluate_genericClf import extract_feature_to_csv

pydub.AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

def split_file_to_short_wav(fileName):
    """
    spits wav file to chunks of audio files that are atleast  the min(file_duration,3 sec)
    :param fileName: the file that being taken to examination
    :return: list of the chun
    """
    # Load your audio.
    song = AudioSegment.from_wav(fileName)
    """
        # Split track where the silence is 3 seconds or more and get chunks using
        # the imported function.
    """
    if song.duration_seconds <= 5:
        chunks=[song]
    else:
        # song = match_target_amplitude(song, -30.0)
        # split the file by silence according to the silence tresh- may vary from file to file
        chunks = split_on_silence(song, min_silence_len=2000, silence_thresh=-25, keep_silence=100)
    """
        merge neighbour chunks that their length is shorter than 3 sec (if there is more than one chunk)  
    """
    target_length = 3 * 1000
    if chunks.__len__() == 0:
        output_chunks = song
    else:
        output_chunks = [chunks[0]]
    for chunk in chunks[1:]:
        if len(output_chunks[-1]) < target_length:
            output_chunks[-1] += chunk
        else:
            # if the last output chunk is longer than the target length,
            # we can start a new one
            output_chunks.append(chunk)
    """
        # Process each chunk with your parameters
    """
    chunk_to_process=[]
    file_name_without_path=os.path.basename(fileName)
    fileName = file_name_without_path.replace(" ", "_")  # lev bug fix to align csv columns - saved me the search
    for i, chunk in enumerate(output_chunks):
        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=500)

        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk

        # Normalize the entire chunk.
        normalized_chunk = match_target_amplitude(audio_chunk, -20.0)

        # Export the audio chunk with new bitrate.
        print("Exporting chunk{0}.wav.".format(i))
        chunk_path= "examined_files//"+str(fileName)+"_chunk_{0}".format(i)+".wav"
        chunk_name= str(fileName)+"_chunk_{0}".format(i)+".wav"
        normalized_chunk.export(
            chunk_path,
            bitrate = "192k",
            format = "wav"
        )
        curr_chunk = [chunk, chunk_name, chunk_path]
        chunk_to_process += [curr_chunk]
    """
        return a list where each node contains [the chunk itself, the chunk's name, the chunk's location ]
    """
    return chunk_to_process

def write_data_aux(chunks_l):
    """
    writes each chunks features to csv to each mfcc size (12,15,20)
    :param chunks_l: a list where each node contains [the chunk itself, the chunk's name, the chunk's location ]
    :return: ----
    """
    for chunk in chunks_l:
        for i in [12,15,20]:
            clfGlobals.change_mfcc_size(12)
            create_csv_different_mfcc()
            extract_feature_to_csv(chunk[1], chunk[2], "test", "csv/test/data_test_mfcc_"+str(i)+".csv", 0, i)


def extract_feature_to_csv(wav_name,wav_path, label, data_file_path, min_wav_duration, fcc_amount):
    """

    :return: writes one row to wav_path with extracted features

    """
    wav_data, sampling_rate = librosa.load(wav_path, duration=5)

    wav_duration = librosa.get_duration(y=wav_data, sr=sampling_rate)

    # lev- dont use really short audio
    if (wav_duration < min_wav_duration):
        print("skipping " + wav_path + " ,duration= " + str(wav_duration))
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
        to_append = f'{str(wav_name)} {np.mean(feature_wav_spec_cent)} {np.mean(zcr)} {np.mean(rolloff)} {np.mean(chroma_stft)}' \
                    f' {np.mean(rms)} {np.mean(mel_spec)}'
        # line = [np.mean(feature_wav_spec_cent), np.mean(zcr), np.mean(rolloff), np.mean(chroma_stft), np.mean(rms),  np.mean(mel_spec)]
        for e in mfccs:
            to_append += f' {np.mean(e)}'
            # line.append(np.mean(e))
        to_append += f' {label}'
        # line=np.array([line])
        #  save to csv (append new lines)
        file = open(data_file_path, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

# the same as lev did with a bit more static variables
class global_For_Clf():
    def __init__(self, clf_label):
        #  changed for every class (for example: scream, cry, ...)
        self.clf_label = clf_label  # have to create a clf with a label
        self.n_mfcc = 12  # lev's initial value here was 40- this is the feature resolution- usually between 12-40

        # keeping the hardcoded 20 mfcc below until end of project submission, later update it to generic mfcc amount
        self.data_file_path = 'csv/'+str(self.get_clf_label())+'/data_'+str(self.get_clf_label())+'_mfcc_'+str(self.n_mfcc)+'.csv'  # cry
        self.csv_to_pkl_path = 'pickle/'+str(self.get_clf_label())+'/combined_lower_amount.pkl' # relevant to modular file TODO currently this is only for scream
        self.path_csv_train_test_data = 'csv/'+str(self.get_clf_label())+'/train_test_data.csv'  # chosen 1:1 ratio data, selected from data.csv
        self.resultsPath = 'results/'+str(self.get_clf_label())+'/experiments_results.csv'
        self.predictor_data = 'predictor_data.csv'
        # end of class changes

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
        self.bestModelsPath = 'models/best_from_final_models'
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

    def change_mfcc_size(self, mfcc_size):
        self.n_mfcc = mfcc_size
        self.data_file_path = 'csv/'+str(self.get_clf_label())+'/data_'+str(self.get_clf_label())+'_mfcc_'+str(self.n_mfcc)+'.csv'  # cry
        return

    def get_clf_label(self):
            return self.clf_label


def create_csv_different_mfcc():
    """
    input: creates csv file for the size of the mfcc needed if it exists does nothing
    """
    # important variables
    data_file_path = clfGlobals.data_file_path
    min_wav_duration = clfGlobals.min_wav_duration
    #  print(data_file_path, min_wav_duration)

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
            # print(f'OK: {len_actual_mfcc_features} ==  n_mfcc_number={n_mfcc_number}')
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

def create_predictor_csv():
    """
    input: creates csv file for the predictor results for each chunk
    output: .csv file with the appropriate columns of nothing if it's already exists
    """
    # important variables
    data_file_path = 'predictor_data.csv'

    if os.path.exists(data_file_path):
        return

    # create header for csv
    header = 'date filename length(sec)'
    directory_path = 'train/positive'
    list_of_files = listdir(directory_path)
    for i in list_of_files:
        header += f' label_{i}'
    header += ' score'
    header += ' label'
    header = header.split()  # split by spaces as default

    file = open(data_file_path, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)


def predictor_to_all_file_chunks(chunks):
    """
    writes predictions for file's chunks
    :param chunks:a list where each node contains [the chunk itself, the chunk's name, the chunk's location ]
    :return: print predictions for each file and count results for all the chunks of the file
    """
    count_positives = 0
    for chunk in chunks:
        current_result, labels_found = predictor_results_to_csv(chunk[1], chunk[0].duration_seconds)
        print("Labels that was found in that chunck are: "+str([label for label in labels_found if label != '0']))
        count_positives +=current_result
    return count_positives

def predictor_results_to_csv(wav_name, wav_duration):
    """
    :param wav_name: file's name
    :param wav_duration: file's duration
    :return: writes one row to wav_path with extracted features

    """

    """
    # search for the models the needed to be loaded by the types of labels we trained for - models list
    also save their sizes in order to insert the right input later on
    """
    print("the chunk "+wav_name+" is being examined for disterss")
    directory_path = 'train/positive'
    list_of_files = listdir(directory_path)
    models = []
    models_mfcc_sizes = []
    for i in list_of_files:
        model_path = str(f'{clfGlobals.bestModelsPath}/{i}_clf_mfcc_12.h5')
        if os.path.exists(model_path):
            # print(f"loading model from {model_path}")
            models_mfcc_sizes.append(12)
            models.append(load_model(model_path))
            continue
        model_path = str(f'{clfGlobals.bestModelsPath}/{i}_clf_mfcc_15.h5')
        if os.path.exists(model_path):
            # print(f"loading model from {model_path}")
            models_mfcc_sizes.append(15)
            models.append(load_model(model_path))
            continue
        model_path = str(f'{clfGlobals.bestModelsPath}/{i}_clf_mfcc_20.h5')
        if os.path.exists(model_path):
            # print(f"loading model from {model_path}")
            models_mfcc_sizes.append(20)
            models.append(load_model(model_path))
            continue
    """
    # search for the right row in each csv file of the features and load it to dict for further use
    """
    # input file name you want to search
    name = wav_name
    # read csv, and split on "," the line
    csv_file_12 = csv.reader(open('csv/test/data_test_mfcc_12.csv', "r"), delimiter=",")
    csv_file_15 = csv.reader(open('csv/test/data_test_mfcc_15.csv', "r"), delimiter=",")
    csv_file_20 = csv.reader(open('csv/test/data_test_mfcc_20.csv', "r"), delimiter=",")
    # loop through csv list
    dict = {}
    for row in csv_file_12:
        # if current rows 2nd value is equal to input, save that row
        if name == row[0]:
            dict[12] = row
            dict[12].pop(0)
            dict[12].pop()
    for row in csv_file_15:
        # if current rows 2nd value is equal to input, save that row
        if name == row[0]:
            dict[15] = row
            dict[15].pop(0)
            dict[15].pop()
    for row in csv_file_20:
        # if current rows 2nd value is equal to input, save that row
        if name == row[0]:
            dict[20] = row
            dict[20].pop(0)
            dict[20].pop()
    """
        # the first features to be written in the predictor csv is the file name date and length of the file
    """
    now = datetime.datetime.now()
    to_append = f'{str(str(now.day)+"-"+str(now.month)+"-"+str(now.year)+":"+str(now.hour)+"-"+str(now.minute)+"-"+str(now.second))} {str(wav_name)} {str(wav_duration)} '
    """
        # for each label insert the correct input , append the result to the row string and count the number of positive
         predictions found so far (count_positives)
    """
    count_positives=0
    labels_that_appeared=list_of_files
    for i, m in enumerate(models, 0):
        # the input that is right for the model
        X=np.array(dict[models_mfcc_sizes[i]])
        scaler = StandardScaler()
        # the input after being normalized - an input with 1 column and 18,21,26 rows is recieved
        X_test_scaled = scaler.fit_transform(X[:, np.newaxis])
        # in order the insert the input to the model we nedd to inverse it to a row with 18,21,26 inputs
        X_inverse = X_test_scaled.transpose()
        # print(np.max(X_test))  # 2590
        # print(np.max(X_inverse))  # 5
        # this bring a 0/1 answer
        # prediction=m.predict(X_inverse)
        # this bring a probability- float number between 0 to 1
        # the next line is the one who gets the results of the label classifier
        prediction=m.predict_proba(X_inverse)
        result1=np.sum(prediction[0])
        # result2=np.argmax(prediction[0])
        # print(result1)
        # print(result2)
        # current_label_score2=m.predict_classes(input_for_label)
        # print(current_label_score[0][0])
        # if the probabilty is over 0.5 than round to 1 otherwise 0
        current_label_score = 1 if result1 >= 0.5 else 0
        # current_label_score = prediction[0]
        # current_label_score2=current_label_score2.shape[0]
        to_append += f' {current_label_score}'
        count_positives += current_label_score
        # remove the label from the label list if its not found positive from that label
        labels_that_appeared[i] = 0 if current_label_score == 0 else labels_that_appeared[i]
    # sum of the labels results
    to_append += f' {count_positives}'
    # if we found at leaset one positive label than we categorize the sound as a distress
    to_append += f' {(count_positives>0)}'

    #  save to csv (append new lines)
    file = open('predictor_data.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())

    # return the chunk result in order to sum all the results of the chunks of the file
    return (count_positives>0), [str(label) for label in labels_that_appeared]



if __name__ == "__main__":
    clfGlobals = global_For_Clf("test")  # create global variable
    create_predictor_csv()
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    # wav_name = input()
    # copy the file to a new location before the process begins
    shutil.copy2(file_path, "source_files")
    chunk_to_process = split_file_to_short_wav(file_path)
    write_data_aux(chunk_to_process)
    count_positives=predictor_to_all_file_chunks(chunk_to_process)
    print("The file is found" + str(" positive " if count_positives > 0 else " negative ")+"for a distress")






