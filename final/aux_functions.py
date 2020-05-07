from pathlib import Path

import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np


def save_scaled_scaler():
    # latest csv path
    for mfcc_size in [12,15,20]:
        csv_path = f'csv/whisper/data_experiment_2_mfcc_{mfcc_size}.csv'

        data_csv = pd.read_csv(csv_path)
        data_no_fileName = data_csv.drop(['filename'], axis=1)
        only_features = np.array(data_no_fileName.iloc[:, :-1], dtype=float)

        scaler = StandardScaler()
        scaler.fit(only_features)
        scaler_filenPath = f'saved_scalers/scaler_mfcc_{mfcc_size}.save'
        joblib.dump(scaler, scaler_filenPath)

def load_scaler_and_test():
    for mfcc_size in [12,15,20]:
        csv_path = f'csv/whisper/data_experiment_2_mfcc_{mfcc_size}.csv'
        scaler_filenPath = f'saved_scalers/scaler_mfcc_{mfcc_size}.save'
        scaler = joblib.load(scaler_filenPath)
        # load csv
        data_csv = pd.read_csv(csv_path)
        data_no_fileName = data_csv.drop(['filename'], axis=1)
        only_features = np.array(data_no_fileName.iloc[:, :-1], dtype=float)
        print(only_features[0,:])
        print("and now scaled: ")
        scaled = scaler.transform(only_features)
        print(scaled[0, :])

def test_sort():
    path_test = Path("examined_files")
    wave_file_paths = sorted(path_test.glob('**/*.wav'))  # <class 'generator'>
    to_sort = []
    for file in wave_file_paths:
        print(file)
        to_sort.append(str(file))

    print("now sorted:")
    print(sorted(to_sort,key=len))


if __name__ == "__main__":
    # save_scaled_scaler()
    #load_scaler_and_test()
    test_sort()


