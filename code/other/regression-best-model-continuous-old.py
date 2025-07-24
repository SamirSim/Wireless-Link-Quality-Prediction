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
from tensorflow.keras.layers import LSTM, Dense # type: ignore
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

PREDICTION_WINDOW = 20 # Number of predictions ahead to make
PREDICTION_WINDOW_step = 1 # To be used in the for loop

def create_sequences(data, window_step):
    X, y = [], []
    for i in range(len(data) - window_step):
        a = data[i:(i + window_step)]
        X.append(a)
        y.append(data[i + window_step])
    return np.array(X), np.array(y)

def evaluate_model(model, x, y):
    X_train, y_train = x, y
    model.fit(X_train, y_train)

    index = 0
    res = {}
    for i in range(1, PREDICTION_WINDOW+1, PREDICTION_WINDOW_step):
        res[i] = []

    while index < len(y_train):
        # Make PREDICTION_WINDOW predictions
        input_window = X_train[index]
        #print("input window: ", input_window)
        for i in range(1, PREDICTION_WINDOW+1, PREDICTION_WINDOW_step):
            if index + i > len(y_train):
                break
            # Predict the next value
            reshaped_input_window = np.array(input_window).reshape(1, -1)
            predicted_value = model.predict(reshaped_input_window)
            res[i].append(predicted_value[0])
        
            # Create a list from the last window_step-1 elements of the input_window
            new_list = []
            new_list = input_window[1:window_step]
            input_window = np.append(new_list, predicted_value[0])
        index = index + 1

    for key in res.keys():
        pass
        #print("key: ", key, " length: ", len(res[key]), " elements: ", res[key])

    error = 0
    for value in res.values():
        mse = mean_squared_error(y_train[-len(value):], value)
        error = error + (mse * mse)

    error = error / PREDICTION_WINDOW
        
    return error

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
#couples = [(2, 10), (2, 9), (7, 9), (7, 6), (10, 6)]
#couples = [(5,6)]
k = -1

r_squared_list = []
rmse_list = []

best_cfgs = {}


randomForestRegressor = RandomForestRegressor()
linearRegressor = LinearRegression()
svr = SVR(kernel='rbf')
gradientBoostingRegressor = GradientBoostingRegressor()
decisionTreeRegressor = DecisionTreeRegressor()
adaBoostRegressor = AdaBoostRegressor()
extraTreesRegressor = ExtraTreesRegressor()
kNeighborsRegressor = KNeighborsRegressor()
lstm = Sequential()

regressors = [randomForestRegressor, svr, gradientBoostingRegressor, decisionTreeRegressor, adaBoostRegressor, extraTreesRegressor]
#regressors = [svr]
regressor_strings = [str(regressor) for regressor in regressors]

for n, m in couples:
    k = k + 1
    #print(k)
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

    min_error = float("inf")

    for regressor_string in regressor_strings:
        window_steps = [3, 5, 10, 15, 20]
        for window_step in window_steps:
            # Prepare sequences
            X_train, y_train = create_sequences(train, window_step)
            
            regressor = RandomForestRegressor() # Default
            if 'Sequential' in regressor_string: # LSTM
                regressor = Sequential()
                # Normalize data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_train = scaler.fit_transform(np.array(train).reshape(-1, 1))
                scaled_test = scaler.transform(np.array(test).reshape(-1, 1))

                X_train, y_train = create_sequences(scaled_train, window_step)
                X_test, y_test = create_sequences(scaled_test, window_step)
                
                # Reshape input to be [samples, time steps, features]
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                regressor.add(LSTM(50, return_sequences=True, input_shape=(window_step, 1)))
                regressor.add(LSTM(50, return_sequences=False))
                regressor.add(Dense(25))
                regressor.add(Dense(1))
                regressor.compile(optimizer='adam', loss='mean_absolute_error')
            elif "RandomForestRegressor" in regressor_string:
                regressor = RandomForestRegressor()
            elif "LinearRegression" in regressor_string:
                regressor = LinearRegression()
            elif "SVR" in regressor_string:
                regressor = SVR()
            elif "GradientBoostingRegressor" in regressor_string:
                regressor = GradientBoostingRegressor()
            elif "DecisionTreeRegressor" in regressor_string:
                regressor = DecisionTreeRegressor()
            elif "AdaBoostRegressor" in regressor_string:
                regressor = AdaBoostRegressor()
            elif "ExtraTreesRegressor" in regressor_string:
                regressor = ExtraTreesRegressor()
            elif "KNeighborsRegressor" in regressor_string:
                regressor = KNeighborsRegressor()

            error = evaluate_model(regressor, X_train, y_train)

            #print(regressor, window_step, error)
            #time.sleep(1)
            if error < min_error:
                min_error = error
                best_regressor = regressor
                best_window_step = window_step

    best_cfgs[key] = {"regressor": str(best_regressor), "window_step": best_window_step, "error": min_error}

print(best_cfgs)
#time.sleep(2)

res_final = {}
for n, m in couples:
    k = k + 1
    result = {}
    #print(k)
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

    best_regressor = best_cfgs[key]["regressor"]
    best_window_step = best_cfgs[key]["window_step"]
    min_error = best_cfgs[key]["error"]
        
    # The best regressor and window step are found
    print("couple: ", (n,m), " Best regressor: ", best_regressor, " window step: ", best_window_step, " error: ", min_error)
    
    window_step = best_window_step
    regressor_string = str(best_regressor)
    if 'Sequential' in regressor_string: # LSTM
        regressor = Sequential()
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(np.array(train).reshape(-1, 1))
        scaled_test = scaler.transform(np.array(test).reshape(-1, 1))
        X_train, y_train = create_sequences(scaled_train, window_step)
        X_test, y_test = create_sequences(scaled_test, window_step)
                    
        # Reshape input to be [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        regressor.add(LSTM(50, return_sequences=True, input_shape=(window_step, 1)))
        regressor.add(LSTM(50, return_sequences=False))
        regressor.add(Dense(25))
        regressor.add(Dense(1))
        regressor.compile(optimizer='adam', loss='mean_absolute_error')
    elif "RandomForestRegressor" in regressor_string:
        regressor = RandomForestRegressor()
    elif "LinearRegression" in regressor_string:
        regressor = LinearRegression()
    elif "SVR" in regressor_string:
        regressor = SVR()
    elif "GradientBoostingRegressor" in regressor_string:
        regressor = GradientBoostingRegressor()
    elif "DecisionTreeRegressor" in regressor_string:
        regressor = DecisionTreeRegressor()
    elif "AdaBoostRegressor" in regressor_string:
        regressor = AdaBoostRegressor()
    elif "ExtraTreesRegressor" in regressor_string:
        regressor = ExtraTreesRegressor()
    elif "KNeighborsRegressor" in regressor_string:
        regressor = KNeighborsRegressor()

    X_train, y_train = create_sequences(train, window_step)
    #print("train sequence: ", X_train, y_train)
    #print("test sequence: ", X_test, y_test)
    #time.sleep(1)
    regressor.fit(X_train, y_train)

    test = train[-window_step:] + test

    X_test, y_test = create_sequences(test, window_step)

    #print(X_test, y_test)

    index = 0
    res = {}
    for i in range(1, PREDICTION_WINDOW+1, PREDICTION_WINDOW_step):
        res[i] = []

    while index < len(X_test):
        # Make PREDICTION_WINDOW predictions
        input_window = X_test[index]
        #print("input window: ", input_window)
        for i in range(1, PREDICTION_WINDOW+1, PREDICTION_WINDOW_step):
            if index + i > len(y_test):
                break
            # Predict the next value
            reshaped_input_window = np.array(input_window).reshape(1, -1)

            predicted_value = regressor.predict(reshaped_input_window)
            #print("input window: ", reshaped_input_window, " predicted value: ", predicted_value)
            res[i].append(predicted_value[0])
            
            # Create a list from the last window_step-1 elements of the input_window
            new_list = []
            new_list = input_window[1:window_step]
            input_window = np.append(new_list, predicted_value[0])

        refit_data = test[index:index+window_step+1]
        x_fit, y_fit = create_sequences(refit_data, window_step)
        X_train = np.append(X_train, x_fit, axis=0) # You can't just refit with the new data, you must combine it with the old data and refit
        y_train = np.append(y_train, y_fit, axis=0)
        regressor.fit(X_train, y_train)
        #print("Refitting with train appended with: ", x_fit, " y_test: ", y_fit)
        #time.sleep(2)
        index = index + 1

    #print(res)

    error = 0
    for key_, value in res.items():
        mse = mean_squared_error(y_train[-len(value):], value)
        mae = mean_absolute_error(y_train[-len(value):], value)
        print(key_, " MSE: ", mse, " MAE: ", mae)
        result[key_] = {"mse": mse, "mae": mae}
    
    res_final[key] = result

print(res_final)
#sys.exit(0)
# Save the results to a file
with open('../data/best-model-continuous.json', 'w') as file:
    json.dump(res_final, file)