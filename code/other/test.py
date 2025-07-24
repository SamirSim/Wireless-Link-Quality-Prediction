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

PREDICTION_WINDOW = 10 # Number of predictions ahead to make
PREDICTION_WINDOW_step = 3 # To be used in the for loop
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
    print("X_train: ", X_train, " y_train: ", y_train)
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

    #print(res)

    for key in res.keys():
        pass
        #print("key: ", key, " length: ", len(res[key]), " elements: ", res[key])

    error = 0
    for value in res.values():
        mse = mean_squared_error(y_train[-len(value):], value)
        error = error + (mse * mse)

    error = error / PREDICTION_WINDOW
        
    return error

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
couples = [(5,6)]

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

#regressors = [randomForestRegressor, svr, gradientBoostingRegressor, decisionTreeRegressor, adaBoostRegressor, extraTreesRegressor]
regressors = [svr]
regressor_strings = [str(regressor) for regressor in regressors]

for n, m in couples:
    k = k + 1
    #print(k)
    sender = "m3-"+str(n)
    receiver = "m3-"+str(m)

    key = sender+"_"+receiver

    data_expe_values = [x for x in range(0, 40)]
            
    # split into train and test sets
    X = data_expe_values
    size = int(len(X) * 0.66) # First 2/3 of the data are used for training, and the rest is used for testing
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = []

    min_error = float("inf")
    res = {}
    for regressor_string in regressor_strings:
        min_error_regressor = float("inf")
        window_steps = [5, 10]
        for window_step in window_steps:
            # Prepare sequences
            X_train, y_train = create_sequences(train, window_step)
            
            regressor = None # Default
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

            print(regressor, window_step, error)
            #time.sleep(1)

            if error < min_error_regressor:
                min_error_regressor = error
                best_window_step_regressor = window_step

        res[regressor_string] = {"window_step": best_window_step_regressor, "error": min_error_regressor}
    best_cfgs[key] = res

#print(best_cfgs)
time.sleep(2)

res_final = {}
for n, m in couples:
    k = k + 1
    result = {}
    #print(k)
    sender = "m3-"+str(n)
    receiver = "m3-"+str(m)

    key = sender+"_"+receiver

    # split into train and test sets
    X = data_expe_values
    size = int(len(X) * 0.66) # First 2/3 of the data are used for training, and the rest is used for testing
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = []

    errors_list = {}
    prediction_list = {}

    for regressor_string in regressor_strings:
        window_step = best_cfgs[key][regressor_string]["window_step"]
        regressor = None # Default
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

        #print(X_train, y_train)
        regressor.fit(X_train, y_train)
        #print("Fitted model: ", regressor, " with train: ", X_train, " y_train: ", y_train)

        errors_list[regressor] = []
        prediction_list[regressor] = []

        train, test = X[0:size], X[size:len(X)]
        test = train[-window_step:] + test

        X_test, y_test = create_sequences(test, window_step)

        last_prediction_list = train[-HISTORY_WINDOW+1:]
        last_y = np.append(train[-HISTORY_WINDOW+1:], test[window_step])

        #print(X_test, y_test, last_prediction_list, last_y)
        #time.sleep(1)

        index = 0
        res = {}
        for i in range(1, PREDICTION_WINDOW+1, PREDICTION_WINDOW_step):
            res[i] = []

        next_predictions = {}

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
                #print("input window: ", reshaped_input_window, " predicted value: ", predicted_value)
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
            #time.sleep(1)
            index = index + 1

            # This part is to calculate the error for each model for each prediction iteratively at each interval, for the last PREDICTION_WINDOW predictions
            #print("Compare between prediction list: ", last_prediction_list, " last_y: ", last_y)
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
        next_predictions[regressor] = res

    #print(next_predictions)
    #time.sleep(5)

    # Find the succession of best models for the adaptive method
    best_models = []
    best_errors = []
    best_predictions = []

    #print("prediction list: ", prediction_list)

    for i in range(0, len(y_test)):
        best_error = sys.maxsize
        for key_, value in errors_list.items():
            #print("Model: ", key_, " Error: ", value[i])
            if value[i] < best_error:
                best_error = value[i]
                best_model = key_
        best_models.append(best_model)
        best_errors.append(best_error)
        best_predictions.append(prediction_list[best_model][i])

    #print(best_models)
    #print(best_errors)
    #print("Best predictions: ", best_predictions)
    #time.sleep(5)

    final_elem = {}
    for i in range(1, PREDICTION_WINDOW+1, PREDICTION_WINDOW_step):
        final_elem[i] = []

    for l in best_predictions:
        i = 1
        for v in l:
            final_elem[i].append(v)
            i = i + PREDICTION_WINDOW_step

    #print(final_elem)
    #time.sleep(2)

    size = len(y_test)
    for key_, value in final_elem.items():
        print("Key: ", key_, " Value in final elem: ", value, " y_test: ", y_test[-(size-key_+1):])
        time.sleep(2)
        mse = mean_squared_error(y_test[-(size-key_+1):], value)
        mae = mean_absolute_error(y_test[-(size-key_+1):], value)
        result[key_] = {"mse": mse, "mae": mae}

    #print("Result: ", result)
    print(result)
    res_final[key] = result

print(res_final)

# Save the results to a file
with open('../data/adaptive-model-continuous.json', 'w') as file:
    json.dump(res_final, file)