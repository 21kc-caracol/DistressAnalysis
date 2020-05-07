from keras import models
from keras import layers


from pathlib import Path

import pickle as pkl


from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler


from keras.layers.core import Dense, Dropout, Activation

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

def create_model2(layers_l, batch_size):
    model = models.Sequential()
    model.add(
        layers.Dense(units=batch_size, activation='relu', input_shape=(X_train.shape[1],)))
    for i in range(layers_l):
        # if i == 0:
        #     # TODO: Add a dense layer
        #     # model.add(Activation(activation))
        # else:
        batch_size=2*batch_size
        model.add(layers.Dense(units=batch_size, activation='relu'))
        # TODO: Add a Dense later AND activation (see above)

    # TODO: Add last dense layer # Note: no activation beyond this point
    model.add(layers.Dense(1, activation='sigmoid'))  # the 1 means binary classification
    model.compile(optimizer='adam'
                  , loss='binary_crossentropy'
                  , metrics=['accuracy'])
    print("in iteration\n")
    print(layers_l, " ", " ", batch_size)
    return model


def create_model(X_train, Y_train, X_test, Y_test):
    model = models.Sequential()
    model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu', input_shape=(X_train.shape[1],)))
    choices={{choice(['one', 'two', 'three', 'four', 'five'])}}
    if choices == 'two':
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
    elif choices == 'three':
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
    elif choices == 'four':
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
    elif choices == 'five':
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # the 1 means binary classification
    model.compile(optimizer='adam'
                  , loss='binary_crossentropy'
                  , metrics=['accuracy'])
    model.fit(X_train, Y_train,
              batch_size={{choice([128, 256, 512])}},
              nb_epoch=20,
              verbose=0,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def create_final_model(X_train, Y_train, X_test, Y_test):
    model = models.Sequential()
    model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu', input_shape=(X_train.shape[1],)))
    # choices={{choice(['one', 'two', 'three', 'four', 'five'])}}
    choices = {{choice(['four'])}}  # based on prior checks
    if choices == 'two':
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
    elif choices == 'three':
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
    elif choices == 'four':
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
    elif choices == 'five':
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # the 1 means binary classification
    model.compile(optimizer='adam'
                  , loss='binary_crossentropy'
                  , metrics=['accuracy'])
    model.fit(X_train, Y_train,
              batch_size={{choice([128, 256, 512])}},
              nb_epoch=20,
              verbose=0,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}



def best_result_search():
    """"
    important to use scaling only after splitting the data into train/validation/test
    scale on training set only, then use the returend "fit" parameters to scale validation and test
    """

    # keras
    #  how i expect to receive a model:
    # loaded model = jsonLoad \pandaLoad mori's choice on the saving format

    # model = models.Sequential()
    #
    # # print(X_train_kfold_scaled.shape[1])  #  45 (is the number of columns for each sample)
    # model.add(layers.Dense(200, activation='relu', input_shape=(X_train.shape[1],)))
    #
    # # model.add(layers.Dense(128, activation='relu'))
    #
    # model.add(layers.Dense(400, activation='relu'))
    #
    # model.add(layers.Dense(1, activation='sigmoid'))  # the 1 means binary classification
    #
    # model.compile(optimizer='adam'
    #               , loss='binary_crossentropy'
    #               , metrics=['accuracy'])
    #
    # # train model on training set of Kfold
    # history = model.fit(X_train,
    #                     y_train,
    #                     epochs=20,
    #                     batch_size=128)
    #
    # test_loss, test_acc = model.evaluate(X_test, y_test)
    #
    # print('test_acc in fold number : ', test_acc)
    #
    # results = model.evaluate(X_test, y_test)
    # print(f'results on the test data in fold number : ', results)

    model = KerasRegressor(build_fn=create_model, verbose=0)

    # layers = [[30], [20, 40], [15, 30, 40]]
    layers_sizes = [list(range(x)) for x in range(1, 2)]
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # activations = ['relu', 'softmax']
    param_grid = dict(layers_l=list(range(5)) , batch_size=batch_sizes, epochs=[20,30])
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error')

    grid_result = grid.fit(X_test, y_test)

    print([grid_result.best_score_, grid_result.best_params_])

    for scores in grid_result.cv_results_:
        print("%f (%f)" % (scores.mean(), scores.std()))

    print(grid_result.best_params_)


def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
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

def data_func():
    """
    note i've kept the original param's names for easier origin tracking.
    :param X_for_k_fold:  it is wwritten "k_fold" to understand easier the origin i've took it from
    :param X_test:
    :param y_for_k_fold:
    :param y_test:
    :return: normalized data for the hyper parameters search
    """
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    path = Path('pickle/hyperParamSearch')
    current_path_x_train = path / "x_train.pkl"
    with current_path_x_train.open('rb') as file:
        X_train = pkl.load(file)

    current_path_x_test = path / "X_test.pkl"
    with current_path_x_test.open('rb') as file:
        X_test = pkl.load(file)

    current_path_y_train = path / "y_train.pkl"
    with current_path_y_train.open('rb') as file:
        Y_train = pkl.load(file)

    current_path_y_test = path / "y_test.pkl"
    with current_path_y_test.open('rb') as file:
        Y_test = pkl.load(file)

    # scale data
    scaler = StandardScaler()
    scaler.fit(X_train)  # must call fit before calling transform.fitting on train, using on train+test+valid
    X_train = scaler.transform(X_train)
    # print(np.amax(X_train_kfold))  # 9490.310668945312
    # print(np.amax(X_train_kfold_scaled))  # 8.236592246485245
    X_test = scaler.transform(X_test)

    return X_train, Y_train, X_test, Y_test

def save_data_to_pickle(X_train, X_test, y_train, y_test):
    """
    save test data as pkl at specific folder
    while overriding the data there - it is ok .
    """

    path_x_test = Path(f"pickle/hyperParamSearch/X_test.pkl")
    with path_x_test.open('wb') as file:
        pkl.dump(X_test, file)
    path_y_test = Path(f"pickle/hyperParamSearch/y_test.pkl")
    with path_y_test.open('wb') as file:
        pkl.dump(y_test, file)

    current_path_x_train = Path(f"pickle/hyperParamSearch/x_train.pkl")
    with current_path_x_train.open('wb') as file:
        pkl.dump(X_train, file)

    current_path_y_train = Path(f"pickle/hyperParamSearch/y_train.pkl")
    with current_path_y_train.open('wb') as file:
        pkl.dump(y_train, file)


def get_optimised_model(X_for_k_fold, X_test, y_for_k_fold, y_test):
    #                                            data=data_func(X_for_k_fold, X_test, y_for_k_fold, y_test)
    save_data_to_pickle(X_for_k_fold, X_test, y_for_k_fold, y_test)
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data_func,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials(),
                                          eval_space=True)
    print(f"best_run= {best_run}")
    return best_model

def get_optimised_model_final(X_for_k_fold, X_test, y_for_k_fold, y_test):
    """
    return: best classifier after hyper parameter search
    receives: split dataset for train and test and classifier label (scream, cry)
    for pickle files directory navigation
    """

    save_data_to_pickle(X_for_k_fold, X_test, y_for_k_fold, y_test)
    best_run_inner, best_model_inner = optim.minimize(model=create_final_model,
                                          data=data_func,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials(),
                                          eval_space=True)
    print(f"best_run= {best_run_inner}")
    best_model_inner.summary()      # print summary
    return best_model_inner


if __name__ == "__main__":
    # load from pickle test data
    # path = Path('module_data')
    # current_path_x_train = path / f"data_for_train_x.pkl"
    # with current_path_x_train.open('rb') as file:
    #     X_train = pkl.load(file)
    #
    # current_path_x_test = path / f"data_for_test_x.pkl"
    # with current_path_x_test.open('rb') as file:
    #     X_test = pkl.load(file)
    #
    # current_path_y_train = path / f"data_for_train_y.pkl"
    # with current_path_y_train.open('rb') as file:
    #     Y_train = pkl.load(file)
    #
    # current_path_y_test = path / f"data_for_test_y.pkl"
    # with current_path_y_test.open('rb') as file:
    #     Y_test = pkl.load(file)


    # best_run, best_model = optim.minimize(model=create_model,
    #                                       data=data,
    #                                       algo=tpe.suggest,
    #                                       max_evals=100,
    #                                       trials=Trials(),
    #                                       eval_space=True)

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data_func,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials(),
                                          eval_space=True)
    X_train, Y_train, X_test, Y_test = data_func()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)



