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

couples = [(166, 163)]

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

    # Load next_predictins from temp.json
    with open('../data/temp.json', 'r') as file:
        next_predictions = json.load(file)

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
        mse = (data_expe[key][0] - next_predictions[model]["1"][0]) ** 2
        if mse < lowest_mse:
            lowest_mse = mse
            selected_model = model

    # Use the best model initially
    data_expe[key] = data_expe[key][-len(next_predictions[selected_model]["1"]):]
    print(len(data_expe[key]), len(next_predictions[selected_model]["1"]))
    # Loop over the remaining time steps
    for i in range(1, len(data_expe[key])):
        current_prediction = next_predictions[selected_model]["1"][i]
        lowest_mse = (data_expe[key][i] - current_prediction) ** 2  # MSE of current model

        # Check if another model performs better
        for model in models:
            candidate_prediction = next_predictions[model]["1"][i]
            mse = (data_expe[key][i] - candidate_prediction) ** 2

            if mse < lowest_mse:  # Switch to a better-performing model
                lowest_mse = mse
                selected_model = model
                current_prediction = candidate_prediction

        # Store the best prediction for this step
        best_predictions.append(current_prediction)

    print("Best predictions sequence:", best_predictions)
    print("Final selected model:", selected_model)
