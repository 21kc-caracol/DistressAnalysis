#  imports
import sys
import librosa
import librosa.display
from os import listdir
from os.path import isfile, join
import csv
import sys

from audioread import NoBackendError
from keras import models
from keras import layers
import numpy as np
import pandas as pd
import datetime

from pydub.utils import make_chunks
from sklearn.externals import joblib
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
from evaluate_genericClf import extract_feature_to_csv

pydub.AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def split_file_to_short_wav(file_path,split_duration):
    """
    spits wav file to chunks of X seconds (3 currently)
    :param fileName: the file that being taken to examination
    :return: list of the chun
    """
    # todo DELETE OLD EXAMINED FILES?


    file_name = Path(file_path).stem  # return only name without extension
    # Load your audio.
    audio_file = AudioSegment.from_wav(file_path)
    # split into X seconds chunks
    chunk_seconds = split_duration
    chunk_length_ms = chunk_seconds * 1000  # pydub calculates in millisec
    audio_file_chunks = make_chunks(audio_file, chunk_length_ms)  # Make chunks of X sec
    # Export all of the individual chunks as wav files-taken from overflow
    # TODO future change to start and end times of chunck

    for i, chunk in enumerate(audio_file_chunks):
        chunk_sec_start = i * clfGlobals.split_by_sec
        chunk_name = f'sec_start_{chunk_sec_start}.wav'
        chunk_path = f'examined_files/{file_name}_{chunk_name}'
        print("exporting ", chunk_path)

        chunk.export(chunk_path, format="wav")



class global_For_Clf():
    def __init__(self, clf_label):
        #  changed for every class (for example: scream, cry, ...)
        self.clf_label = clf_label  # have to create a clf with a label
        self.n_mfcc = 12  # lev's initial value here was 40- this is the feature resolution- usually between 12-40

        # keeping the hardcoded 20 mfcc below until end of project submission, later update it to generic mfcc amount
        self.data_file_path = 'csv/'+str(self.get_clf_label())+'/data_'+str(self.get_clf_label())+'_mfcc_'+str(self.n_mfcc)+'.csv'  # cry #TODO MERGE WITH GENERIC file
        self.csv_to_pkl_path = 'pickle/'+str(self.get_clf_label())+'/combined_lower_amount.pkl' # relevant to modular file TODO currently this is only for scream
        self.path_csv_train_test_data = 'csv/'+str(self.get_clf_label())+'/train_test_data.csv'  # chosen 1:1 ratio data, selected from data.csv
        self.resultsPath = 'results/'+str(self.get_clf_label())+'/experiments_results.csv'
        self.predictor_data = 'predictor_data.csv'
        self.predictionsPath = 'csv/prediction_results'
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
        self.split_by_sec = 3  # split every X seconds
        self.tested_file_name = ''  # name of file for prediction
        self.predictor_pos_percent_condition = 0  # above 0.XX will consider as positive prediction 0 means dont use this

    def getInputDim(self):
        amount = len(self.csv_initial_head.split()) + self.n_mfcc - 1  # -1 because filename isnt a feature
        return amount

    def get_total_samples(self):
        return self.positives + self.negatives

    def get_model_name(self):
        model_name = (type(self.model)).__name__
        return model_name

    def change_mfcc_size_path(self, mfcc_size):
        self.n_mfcc = mfcc_size
        self.data_file_path = 'csv/'+str(self.get_clf_label())+'/data_'+str(self.get_clf_label())+'_mfcc_'+str(self.n_mfcc)+'.csv'  # cry
        return

    def get_clf_label(self):
            return self.clf_label

    def get_prediction_csv_file_path(self):
        return f'{self.predictionsPath}/{self.predictor_data}'

    def get_data_file_path(self):
        return f'csv/'+str(self.get_clf_label())+'/data_mfcc_'+str(self.n_mfcc)+f'_{self.tested_file_name}.csv'

    def update_mfcc(self,mfcc_size):
        self.n_mfcc = mfcc_size



def create_csv_different_mfcc():
    """
    input: creates csv file for the size of the mfcc needed if it exists does nothing
    """
    # important variables
    data_file_path = clfGlobals.get_data_file_path()
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

    # load features from each wav file- put inside the lines below as a function

    # reaching each wav file
    path_test = Path("examined_files")
    wave_file_paths = path_test.glob(f'{clfGlobals.tested_file_name}*.wav')  # <class 'generator'>
    # lev fix for continuous chunks by time instead of sorted by string alphabet( chunk1,chunk11)
    wave_file_paths_sorted = []
    for file in wave_file_paths:
        wave_file_paths_sorted.append(str(file))


    #  print(type(wave_file_paths))
    count = 0  # for progress tracking
    print('covered WAV files: ')

    for wav_path in sorted(wave_file_paths_sorted, key=len):
        #lev fix for sorting
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
            extract_feature_to_csv(wav_path, clfGlobals.get_clf_label(), data_file_path, min_wav_duration, fcc_amount)
        except NoBackendError as e:
            print("audioread.NoBackendError " + "for wav path " + str(wav_path))
            continue  # one file didnt work, continue to next one


def create_predictor_csv(file_path):
    """
    input: creates csv file for the predictor results for each chunk
    output: .csv file with the appropriate columns of nothing if it's already exists
    """
    # important variables

    file_name = (Path(file_path)).stem
    clfGlobals.tested_file_name = file_name
    clfGlobals.predictor_data= f'{file_name}__prediction_results.csv'
    data_file_path = clfGlobals.get_prediction_csv_file_path()
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


def load_models_by_mfcc(mfcc_size):
    """"
    PARAM: mfcc_size - size by which we will load the relevant models.
    RETURNS: all models with <mfcc_size> in their name. for example: mfcc_size=20 will return [scream_20,gasp_20]
    """
    relevant_models= []
    models_path = f'{clfGlobals.bestModelsPath}'
    list_of_clfs = listdir(models_path)
    for model_name in list_of_clfs:
        if str(mfcc_size) in model_name:
            relevant_models.append(model_name)

    return relevant_models


def add_predictions_to_csv(model_name, y_test_predicted):
    #print(model_name)
    #print(y_test_predicted)

    predictor_panda = pd.read_csv(clfGlobals.get_prediction_csv_file_path())

    clf_column = f'label_'+str(model_name.split('_')[0])  # take only <sniff> from <sniff_clf_mfcc_12.h5>
    #print(predictor_panda.loc[:, clf_column])
    pred_array = []
    for prediction in y_test_predicted:
        pred_array.extend(prediction)

    #print(pred_array)
    predictor_panda[clf_column] = pred_array
    #print(predictor_panda.loc[:,clf_column])
    predictor_panda.to_csv(clfGlobals.get_prediction_csv_file_path(), index=False)



def add_formal_data_to_csv():
    # add data to prediction CSV that isn't prediction-oriented. for example: date, filename
    predictor_panda = pd.read_csv(clfGlobals.get_prediction_csv_file_path())
    csv_wav_chunks_path = clfGlobals.get_data_file_path()  # just take something- any valid number
    examined_wav_chunks_panda = pd.read_csv(csv_wav_chunks_path)
    # add filenames
    predictor_panda['filename'] = examined_wav_chunks_panda.loc[:, 'filename']
    # add time
    now = datetime.datetime.now()
    current_time_date = str(now.day)+"_"+str(now.month)+"_"+str(now.year)+"-"+str(now.hour)+":"+str(now.minute)+":"+str(now.second)
    predictor_panda['date'] = current_time_date
    # add length of wave chunks
    predictor_panda['length(sec)'] = clfGlobals.split_by_sec
    #save to csv
    predictor_panda.to_csv(clfGlobals.get_prediction_csv_file_path(), index=False)




def predictor_to_all_file_chunks():    # predict on each CSV file

    for mfcc_size in [12,15,20]:
        #load appropriate scaler and csv
        clfGlobals.update_mfcc(mfcc_size)  # cruucial command for correct csv path
        csv_path = clfGlobals.get_data_file_path()
        scaler_filenPath = f'saved_scalers/scaler_mfcc_{mfcc_size}.save'
        scaler = joblib.load(scaler_filenPath)
        # load csv
        data_csv = pd.read_csv(csv_path)
        data_no_fileName = data_csv.drop(['filename'], axis=1)
        only_features = np.array(data_no_fileName.iloc[:, :-1], dtype=float)
        X_test_scaled = scaler.transform(only_features)

        # load appropriate models according to mfcc
        model_names = load_models_by_mfcc(mfcc_size)
        for model_name in model_names:
            model_path = str(f'{clfGlobals.bestModelsPath}/{model_name}')
            print(f"loading model from {model_path}")
            model = load_model(model_path)
            # only above 0 will use probabilities
            if clfGlobals.predictor_pos_percent_condition == 0:
                y_test_predicted = model.predict_classes(X_test_scaled)
            else:
                y_test_predicted = np.around(model.predict_proba(X_test_scaled), decimals=2)  # using probabilities 0.XX

            add_predictions_to_csv(model_name, y_test_predicted)
    add_formal_data_to_csv()



def show_results():
    """
    open excel with predictions and print high positive results
    :return:
    """
    predictor_panda = pd.read_csv(clfGlobals.get_prediction_csv_file_path())

    labels_path = 'train/positive'  # TODO improve complexity here- dont touch the Disc
    list_of_labels = listdir(labels_path)
    for label in list_of_labels:
        column_label = f'label_{label}'
        above_condition = predictor_panda[predictor_panda[column_label] > clfGlobals.predictor_pos_percent_condition]
        above_condition_specific = above_condition[['filename', column_label]]
        print(f'Labeled as {label}:')
        print(above_condition_specific)

    # open excel with predictions
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    prediction_FullFilePath = Path(f'{current_script_path}/{clfGlobals.get_prediction_csv_file_path()}')
    os.startfile(prediction_FullFilePath)



def create_csv_for_test_files():
    # create csv and extract features
    """
    writes each chunks features to csv to each mfcc size (12,15,20)
    :param
    :return: ----
    """
    for mfcc_size in [12, 15, 20]:
        clfGlobals.change_mfcc_size_path(mfcc_size)
        create_csv_different_mfcc()


if __name__ == "__main__":
    clfGlobals = global_For_Clf("test")  # create global variable
    clfGlobals.predictor_pos_percent_condition = 0.80  # set a level for positive labeling
    split_duration_sec = clfGlobals.split_by_sec

    # TODO move to globals different paths
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    # file_path = f'lev_test_files/1.wav'  # delete this when using dialog box
    shutil.copy2(file_path, "source_files")

    create_predictor_csv(file_path)  # create CSV which holds the predictions
    split_file_to_short_wav(file_path, split_duration_sec)  # split to X sec wave's
    create_csv_for_test_files()

    predictor_to_all_file_chunks()

    show_results()






