import sys
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from pandas import read_csv # type: ignore
import random
from sklearn import linear_model, metrics # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor # type: ignore
from sklearn.tree import DecisionTreeRegressor # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.svm import SVR # type: ignore
from sklearn.neighbors import KNeighborsRegressor # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from scipy.special import softmax # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, GRU, SimpleRNN, Conv1D, Flatten # type: ignore
from statsmodels.tsa.arima.model import ARIMA # type: ignore
from math import sqrt
from pandas import DataFrame # type: ignore
import time
import warnings
import random
import string
import json

warnings.filterwarnings("ignore")

random.seed(10)

BATCH_SIZE = 10
EPOCHS = 10

def evaluate_model(model, x, y):
    # Evaluate the model
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    y_pred = model.predict(X_val)

    print("Predictions: ", y_pred, "y_val: ", y_val)
    for i in range(len(y_pred)):
        for j in range(len(y_pred[i])):
            if y_pred[i][j] < 0:
                y_pred[i][j] = 0
            elif y_pred[i][j] > 50:
                y_pred[i][j] = 50

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    print("Mean Squared Error: ", mse)
    print("Mean Absolute Error: ", mae)

    print("Real values: ", y_val)
    print("Predicted values: ", y_pred)
    print("Mean Squared Error: ", mean_squared_error(y_val, y_pred))
    
    return mse, mae


# LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(20, input_shape=input_shape))
    model.add(Dense(1))
    return model

# Bi-LSTM Model
def build_bilstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(10), input_shape=input_shape))
    model.add(Dense(1))
    return model

# Encoder-Decoder LSTM Model
def build_ed_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(20, input_shape=input_shape))
    model.add(Dense(1))
    return model

# CNN Model
def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(Conv1D(20, kernel_size=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1))
    return model

# GRU Model
def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(128, input_shape=input_shape))
    model.add(Dense(1))
    return model

# RNN Model
def build_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(64, return_sequences=True, input_shape=input_shape))
    model.add(SimpleRNN(20))
    model.add(Dense(1))
    return model

# Load the series_list from the file
with open('../data/series_list_customized_p.json', 'r') as file:
    series_list = json.load(file)

data_expe = series_list[0]
data_simu = series_list[1]

with open('../data/series_list_p0,56.json', 'r') as file:
    series_list = json.load(file)

data_simu_pdr = series_list[1]

# Create a list to hold the couples
couples = []

# Loop through all numbers from 2 to 12
for i in range(2, 13):
    for j in range(2, 13):
        # Skip pairs where the two elements are the same
        if i != j:
            # Add the pair (i, j) to the list
            couples.append((i, j))
Position = range(1,17)

#couples = [(2, 10), (2, 9), (7, 9), (7, 6), (10, 6), (10, 2), (11, 2), (11,6), (4, 5), (4, 6), (5, 6), (5, 4), (6, 4), (6, 5), (6, 10), (6, 9)]
couples = [(2, 10), (2, 9)]
#couples = [(5,6)]
k = -1

for n, m in couples:
    k = k + 1
    sender = "m3-"+str(n)
    receiver = "m3-"+str(m)

    key = sender+"_"+receiver
    data_expe_values = data_expe[key]
    data_simu_values = data_simu[key]
    data_simu_pdr_values = data_simu_pdr[key]

    data_simu_values = data_simu_values[0:len(data_expe_values)] # Cut the simulation data to match the size of the experiments data
    data_simu_pdr_values = data_simu_pdr_values[0:len(data_expe_values)] # Cut the simulation data to match the size of the experiments data

    x = np.arange(1, len(data_simu_values) + 1) 
    y_exp = np.array(data_expe_values)
    y_sim = np.array(data_simu_values)
    y_sim_pdr = np.array(data_simu_pdr_values)

k = -1
mae_list = []
mse_list = []

best_cfgs = {}

for n, m in couples:
    result = {}
    k = k + 1
    sender = "m3-"+str(n)
    receiver = "m3-"+str(m)

    key = sender+"_"+receiver

    data_expe_values = data_expe[key]
    data_simu_values = data_simu[key]
    data_simu_pdr_values = data_simu_pdr[key]
    data_simu_values = data_simu_values[0:len(data_expe_values)] # Cut the simulation data to match the size of the experiments data
    data_simu_pdr_values = data_simu_pdr_values[0:len(data_expe_values)] # Cut the simulation data to match the size of the experiments data
            
    # split into train and test sets
    X = data_expe_values
    size = int(len(X) * 0.66) # First 2/3 of the data are used for training, and the rest is used for testing
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = []

    lstm = Sequential()

    regressors = [lstm]
    regressor_strings = [str(regressor) for regressor in regressors]
    min_mse = float("inf")
    min_mae = float("inf")

    for regressor_string in regressor_strings:
        window_steps = [3, 5, 10, 15, 20]
        for window_step in window_steps:
            # Prepare sequences
            def create_sequences(data, window_step):
                X, y = [], []
                for i in range(len(data) - window_step):
                    a = data[i:(i + window_step)]
                    X.append(a)
                    y.append(data[i + window_step])
                return np.array(X), np.array(y)

            X_train, y_train = create_sequences(train, window_step)

            input_shape = (window_step, 1)
            # Instantiate each model
            #regressor = build_lstm_model(input_shape)
            #regressor = build_bilstm_model(input_shape)
            #regressor = build_ed_lstm_model(input_shape)
            #regressor = build_cnn_model(input_shape)
            #regressor = build_gru_model(input_shape)
            regressor = build_rnn_model(input_shape)
        
            regressor.compile(optimizer='adam', loss='mean_squared_error')

            mse, mae = evaluate_model(regressor, X_train, y_train)

            print("Window Step: ", window_step, " MAE: ", mae, " MSE: ", mse)
            #time.sleep(2)

            if mse < min_mse:
                min_mae = mae
                min_mse = mse
                best_regressor = regressor
                best_window_step = window_step

    best_cfgs[key] = {"regressor": str(best_regressor), "window_step": best_window_step, "MAE": min_mae, "MSE": min_mse}

print(best_cfgs)

steps_list = [1, 2, 3, 5, 7, 10, 15, 20]
steps_list = [1, 2]
elems = []
for prediction_step in steps_list:
    result = {}
    elem = {}

    for n, m in couples:
        k = k + 1
        sender = "m3-"+str(n)
        receiver = "m3-"+str(m)

        key = sender+"_"+receiver

        data_expe_values = data_expe[key]
        data_simu_values = data_simu[key]
        data_simu_pdr_values = data_simu_pdr[key]
        data_simu_values = data_simu_values[0:len(data_expe_values)] # Cut the simulation data to match the size of the experiments data
        data_simu_pdr_values = data_simu_pdr_values[0:len(data_expe_values)] # Cut the simulation data to match the size of the experiments data
            
        # split into train and test sets
        X = data_expe_values
        size = int(len(X) * 0.66) # First 2/3 of the data are used for training, and the rest is used for testing
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]

        regressor_results = []
        for regressor_string in regressor_strings:
            best_regressor = regressor_string
            best_window_step = best_cfgs[key]["window_step"]
            min_mae = best_cfgs[key]["MAE"]
            min_mse = best_cfgs[key]["MSE"]
            
            # The best regressor and window step are found
            print("couple: ", (n,m), " step: ", prediction_step, " Best regressor: ", best_regressor, " window step: ", best_window_step, " MAE: ", min_mae, " MSE: ", min_mse)
        
            window_step = best_window_step
            input_shape = (window_step, 1)
            # Instantiate each model
            #regressor = build_lstm_model(input_shape)
            #regressor = build_bilstm_model(input_shape)
            #regressor = build_ed_lstm_model(input_shape)
            #regressor = build_cnn_model(input_shape)
            #regressor = build_gru_model(input_shape)
            regressor = build_rnn_model(input_shape)
            
            regressor.compile(optimizer='adam', loss='mean_squared_error')

            X_train, y_train = create_sequences(train, window_step)

            test = train[-window_step:] + test

            input_shape = (window_step, 1)

            regressor.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

            inputs = []
            index = 0
            predictions = []

            while index < len(test):
                print("Index: ", index, len(test))
                new_test = [float(v) for v in test]
                try:
                    for i in range(index, index+prediction_step):
                        input_window = new_test[i:window_step+i]
                        reshaped_input_window = np.array(input_window).reshape(-1, 1)

                        #reshaped_input_window = np.array(input_window).reshape(1, -1)
                        inputs.append(reshaped_input_window)
                        #print("reshaped input window: ", reshaped_input_window)
                        #time.sleep(3)
                        predicted_value = regressor.predict(reshaped_input_window)

                        #predicted_value = scaler.inverse_transform(predicted_value)
                        # Inverse transform the predicted value
                        print("predicted value: ", predicted_value)
                        #time.sleep(2)

                        #print("predicted value: ", predicted_value[0][0])
                        predictions.append(float(predicted_value[0][0]))
                        print("New test: ", new_test)
                        # Replace the element in the TestSet with the predicted value
                        new_test[window_step+i] = float(predicted_value[0][0])
                        i = i + 1
                        #time.sleep(2)

                    #print("test: ", test, index, window_step)
                    print(test[index:index+window_step+prediction_step])
                    refit_data = test[index:index+window_step+prediction_step]
                    #print("refit data: ", refit_data)
                                
                    # Prepare data for refitting
                    X_train, y_train = create_sequences(refit_data, window_step)
                    #print("Refit with: X train: ", X_train, "y train: ", y_train)
                    #time.sleep(1)
                    index = index + prediction_step
                            
                    # Refit the model
                    regressor.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print("Exception occured", e, " Index: ", index, len(test))
                    break
            
            print("New Predictions: ", key, predictions)


            mse = mean_squared_error(test[:len(predictions)], predictions)
            mae = mean_absolute_error(test[:len(predictions)], predictions)

            print("Test: ", test)
            #time.sleep(2)

            regressor_results.append({"regressor": str(regressor), "window_step": window_step, "MAE": mae, "MSE": mse, "predictions": predictions})
            mae_list.append(mae)
            mse_list.append(mse)

        result[key] = regressor_results
        #print("Results: ", regressor_results)
        #print("Result: ", result)
        
    elem [prediction_step] = {"results": result}
    elems.append(elem)    
            
    print('Mean Absolute Error (MAE) for step: ', str(prediction_step), ": ", np.mean(mae_list))
    print('Mean Squared Error (MSE): ', np.mean(mse_list))

print("-----------------")
print(elems)

filename = sys.argv[1].split(".")[0]
with open('../data/'+filename+'.json', 'a') as file:
    json.dump(elems, file)