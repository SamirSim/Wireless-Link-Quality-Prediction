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

PREDICTION_WINDOW = 20 # Number of predictions ahead to make
PREDICTION_WINDOW_step = 1 # To be used in the for loop
HISTORY_WINDOW = 5 # Number of last prediction to be used for evaluation to select the best model in the adaptive approach

def classify_series(time_series):
    mean_value = np.mean(time_series)

    if 0 < mean_value < 9:
        return "Bad"
    elif 9 <= mean_value < 32:
        return "Average"
    elif 32 <= mean_value < 37:
        return "Good"
    elif 37 <= mean_value <= 50:
        return "Excellent"
    else:
        return "Out of range"

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

# Generate all possible sender-receiver pairs without "m3"
couples = [(int(sender[3:]), int(receiver[3:]))  for sender, receiver in permutations(node_mac_map.keys(), 2)]

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


# Open the file to read the best configurations
with open('../data/adaptive-model-continuous-24h-config.json', 'r') as file:
    best_cfgs = json.load(file)

res_final = {}
times_cluster = {}
times_cluster["Bad"] = {}
times_cluster["Average"] = {}
times_cluster["Good"] = {}
times_cluster["Excellent"] = {}
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
        
        avg_time = 0
        while index < 100:
            start = time.time()
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
                #print("i: ", i, "input window: ", reshaped_input_window, " predicted value: ", predicted_value)
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
            #print("Refitting with train appended with: ", x_fit, " y_test: ", y_fit)
            end = time.time()
            print("Time taken for prediction: ", end - start, " seconds for link: ", key , " regressor: ", regressor_string)
            avg_time += (end - start)
            #time.sleep(0.1)

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
            
            index = index + 1
            #time.sleep(2)
        avg_time = avg_time / 100
        if classify_series(data_expe_values) == "Bad":
            if key not in times_cluster["Bad"]:
                times_cluster["Bad"][key] = []
            times_cluster["Bad"][key].append(avg_time)
            print("Bad: ", key, " time: ",avg_time)
        elif classify_series(data_expe_values) == "Average":
            if key not in times_cluster["Average"]:
                times_cluster["Average"][key] = []
            times_cluster["Average"][key].append(avg_time)
        elif classify_series(data_expe_values) == "Good":
            if key not in times_cluster["Good"]:
                times_cluster["Good"][key] = []
            times_cluster["Good"][key].append(avg_time)
        elif classify_series(data_expe_values) == "Excellent":
            if key not in times_cluster["Excellent"]:
                times_cluster["Excellent"][key] = []
            times_cluster["Excellent"][key].append(avg_time)
        #time.sleep(1)

        #print("errors_list: ", errors_list)
        #time.sleep(2)
        next_predictions[regressor] = res

    #print("y_test: ", y_test, len(y_test))
    #time.sleep(2)

    print("next_predictions: ", next_predictions)
    #time.sleep(5)

print(times_cluster)
#sys.exit(0)
# Save the results to a file
with open('../data/adaptive-times-clusters.json', 'w') as file:
    json.dump(times_cluster, file)