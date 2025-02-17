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

from itertools import permutations

# Node MAC mapping
node_mac_map = {
    "m3-99": "b277",
    "m3-123": "c276",
    "m3-133": "2360",
    "m3-143": "9779",
    "m3-150": "b676",
    "m3-153": "b081",
    "m3-159": "a081",
    "m3-163": "9276",
    "m3-166": "9671",
}

PREDICTION_WINDOW = 1 # Number of predictions ahead to make
PREDICTION_WINDOW_step = 1 # To be used in the for loop
HISTORY_WINDOW = 5 # Number of last prediction to be used for evaluation to select the best model in the adaptive approach 

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
with open('../data/series-iotj-24h.json', 'r') as file:
    data_expe = json.load(file)

couples = [(150, 163), (133, 153), (166, 163)]

#couples = [(166, 163)]

print(couples, len(couples))

#couples = [(2, 10), (2, 9), (7, 9), (7, 6), (10, 6), (10, 2), (11, 2), (11,6), (4, 5), (4, 6), (5, 6), (5, 4), (6, 4), (6, 5), (6, 10), (6, 9)]
#couples = [(2, 10), (2, 9), (7, 9), (7, 6), (10, 6)]
#couples = [(143,159)]
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

regressors = [randomForestRegressor, svr, gradientBoostingRegressor, decisionTreeRegressor, adaBoostRegressor, extraTreesRegressor]
#regressors = [randomForestRegressor, svr] # TODO Changed here
regressor_strings = [str(regressor) for regressor in regressors]

# Open best configs file
with open('../data/adaptive-model-continuous-24h-config.json', 'r') as file:
    best_cfgs = json.load(file)

print(best_cfgs)

res_final = {}
for n, m in couples:
    k = k + 1
    result = {}
    #print(k)
    sender = "m3-"+str(n)
    receiver = "m3-"+str(m)

    key = sender+"_"+receiver

    data_expe_values = data_expe[key]
            
    # split into train and test sets
    X = data_expe_values
    #X = [x for x in range(0, 80)] # TODO Changed here
    size = int(len(X) * 0.3) # First 2/3 of the data are used for training, and the rest is used for testing
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = []

    errors_list = {}
    prediction_list = {}

    next_predictions = {}
    for regressor_string in regressor_strings:
        #print("regressor_string: ", regressor_string)
        window_step = best_cfgs[key][regressor_string]["window_step"]
        regressor = None # Default
        if "RandomForestRegressor" in regressor_string:
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
        
        #print(X_train, y_train)
        regressor.fit(X_train, y_train)

        errors_list[regressor] = []
        prediction_list[regressor] = []

        train, test = X[0:size], X[size:len(X)]
        test = train[-window_step:] + test

        X_test, y_test = create_sequences(test, window_step)

        last_prediction_list = train[-HISTORY_WINDOW+1:]
        last_y = np.append(train[-HISTORY_WINDOW+1:], test[window_step])

        #print(X_test, y_test, last_prediction_list, last_y)

        index = 0
        res = {}
        for i in range(1, PREDICTION_WINDOW+1, PREDICTION_WINDOW_step):
            res[i] = []
        
        
        while index < len(X_test):
            # Make PREDICTION_WINDOW predictions
            input_window = X_test[index]
            list_predicted_value = []
            #print("input window: ", input_window)
            for i in range(1, PREDICTION_WINDOW+1, PREDICTION_WINDOW_step):
                if index + i > len(y_test):
                    break
                # Predict the next value
                reshaped_input_window = np.array(input_window).reshape(1, -1)

                predicted_value = regressor.predict(reshaped_input_window)
                print("i: ", i, "input window: ", reshaped_input_window, " predicted value: ", predicted_value)
                #time.sleep(1)
                res[i].append(predicted_value[0])

                list_predicted_value.append(predicted_value[0])

                if i == 1:
                    last_prediction_list.append(predicted_value[0])
                
                # Create a list from the last window_step-1 elements of the input_window
                new_list = []
                new_list = input_window[1:window_step]
                input_window = np.append(new_list, predicted_value[0])
            
            prediction_list[regressor].append(list_predicted_value)

            refit_data = test[index:index+window_step+1]
            x_fit, y_fit = create_sequences(refit_data, window_step)
            X_train = np.append(X_train, x_fit, axis=0)
            y_train = np.append(y_train, y_fit, axis=0)
            regressor.fit(X_train, y_train)
            print("Refitting with train appended with: ", x_fit, " y_test: ", y_fit)
            #time.sleep(1)
            index = index + 1

            # This part is to calculate the error for each model for each prediction iteratively at each interval, for the last PREDICTION_WINDOW predictions
            #print("Compare between prediction list: ", last_prediction_list, " last_y: ", last_y, " for model: ", regressor)
            #time.sleep(1)
            error = mean_squared_error(last_y, last_prediction_list)
            errors_list[regressor].append(error)
            #print("Error for model: ", regressor, " is: ", error)
            # Remove the first element of the prediction list
            last_prediction_list = last_prediction_list[1:]
            if index + window_step < len(test):
                last_y = np.append(last_y, test[window_step+index])
            last_y = last_y[1:]
            #time.sleep(2)
        #print("errors_list: ", errors_list)
        #time.sleep(2)
        next_predictions[regressor] = res

    #print("y_test: ", y_test, len(y_test))
    #time.sleep(2)

    print("next_predictions: ", next_predictions)

    best_models = []
    best_errors = []
    best_predictions = []

    # Extract model names
    models = list(next_predictions.keys())

    # Initialize the best predictions list
    best_predictions = []
    selected_model = None

    # Initialize with the best model at the first time step
    lowest_mse = float("inf")

    for model in models:
        mse = (y_test[0] - next_predictions[model][1][0]) ** 2
        if mse < lowest_mse:
            lowest_mse = mse
            selected_model = model

    # Use the best model initially
    y_test = y_test[-len(next_predictions[selected_model][1]):]
    print(len(y_test), len(next_predictions[selected_model][1]))
    
    # Loop over the remaining time steps
    for i in range(1, len(y_test)):
        current_prediction = next_predictions[selected_model][1][i]
        best_predictions.append(current_prediction)

        # Ensure we have enough history before checking for a model switch
        if i >= HISTORY_WINDOW:
            current_model_mse = np.mean([
                (y_test[j] - next_predictions[selected_model][1][j]) ** 2 
                for j in range(i - HISTORY_WINDOW, i)
            ])

            best_new_model = selected_model
            lowest_mse = current_model_mse

            # Check if another model has a better cumulative MSE over the HISTORY_WINDOW
            for model in models:
                if model == selected_model:
                    continue  # Skip the currently selected model

                candidate_mse = np.mean([
                    (y_test[j] - next_predictions[model][1][j]) ** 2 
                    for j in range(i - HISTORY_WINDOW, i)
                ])

                if candidate_mse < lowest_mse:
                    lowest_mse = candidate_mse
                    best_new_model = model

            # Switch models if a better one was found
            if best_new_model != selected_model:
                selected_model = best_new_model

    res_final[key] = best_predictions

print(res_final)

# Save the results to a file
with open ('../data/adaptive-model-continuous-24h-predictions-test.json', 'w') as file:
    json.dump(res_final, file)