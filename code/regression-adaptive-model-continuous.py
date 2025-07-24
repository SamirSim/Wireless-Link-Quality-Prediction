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

FOUND_CONFIG = True # Set to True if the best configurations have already been found and saved in the file, otherwise it will search for the best configurations

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

#couples = [(150, 163)]

#couples = [(99, 123), (99, 133), (99, 143), (99, 150), (99, 153), (99, 159), (99, 163), (99, 166), (123, 99), (123, 133), (123, 143), (123, 150)]
#couples = [(123, 153), (123, 159), (123, 163), (123, 166), (133, 99), (133, 123), (133, 143), (133, 150), (133, 153), (133, 159), (133, 163), (133, 166)]
#couples = [(143, 99), (143, 123), (143, 133), (143, 150), (143, 153), (143, 159), (143, 163), (143, 166), (150, 99), (150, 123), (150, 133), (150, 143)]
#couples = [(150, 153), (150, 159), (150, 163), (150, 166), (153, 99), (153, 123), (153, 133), (153, 143), (153, 150), (153, 159), (153, 163), (153, 166)]
#couples = [(159, 99), (159, 123), (159, 133), (159, 143), (159, 150), (159, 153), (159, 163), (159, 166), (163, 99), (163, 123), (163, 133), (163, 143)]
couples = [(163, 150), (163, 153), (163, 159), (163, 166), (166, 99), (166, 123), (166, 133), (166, 143), (166, 150), (166, 153), (166, 159), (166, 163)]

print(couples, len(couples))

time.sleep(2)

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

if not FOUND_CONFIG:
    for n, m in couples:
        k = k + 1
        print(k)
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

        min_error = float("inf")
        res = {}
        for regressor_string in regressor_strings:
            min_error_regressor = float("inf")
            window_steps = [3, 5, 10, 15, 20] 
            #window_steps = [5] # TODO Changed here
            for window_step in window_steps:
                # Prepare sequences
                X_train, y_train = create_sequences(train, window_step)
                
                regressor = RandomForestRegressor() # Default
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

                error = evaluate_model(regressor, X_train, y_train)

                #print(regressor, window_step, error)
                #time.sleep(1)
                if error < min_error_regressor:
                    min_error_regressor = error
                    best_window_step_regressor = window_step
            
            res[regressor_string] = {"window_step": best_window_step_regressor, "error": min_error_regressor}
        best_cfgs[key] = res

    print(best_cfgs)
    #time.sleep(2)

    with open('../data/adaptive-model-continuous-24h-config.json', 'w') as file:
        json.dump(best_cfgs, file)
else:
    # Load the best configurations from the file
    with open('../data/adaptive-model-continuous-24h-config.json', 'r') as file:
        best_cfgs = json.load(file)

print("Best configurations loaded: ", best_cfgs)

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
    #time.sleep(5)

    best_models = []
    best_errors = []
    best_predictions = []

    #print(PREDICTION_WINDOW+HISTORY_WINDOW, len(y_test)-PREDICTION_WINDOW+1)
    
    # Find the succession of best models for the adaptive method
    for i in range(PREDICTION_WINDOW+HISTORY_WINDOW, len(y_test)+HISTORY_WINDOW+1):
        best_error = sys.maxsize
        val_true = y_test[i-HISTORY_WINDOW-1]
        #print("True value: ", val_true, " i: ", i)
        for regressor, prediction in next_predictions.items():
            error = 0
            for j in range(1, PREDICTION_WINDOW+1, PREDICTION_WINDOW_step):
                print("j: ", j, " i: ", i, " i-HISTORY_WINDOW-j: ", i-HISTORY_WINDOW-j)
                print("Model: ", regressor, " Val: ", prediction[j][i-HISTORY_WINDOW-j], " True value: ", val_true)
                val = prediction[j][i-HISTORY_WINDOW-j]
                mse = mean_squared_error([val_true], [val])
                error = error + (mse * mse)
            error = error / PREDICTION_WINDOW
            if error < best_error:
                best_error = error
                best_model = regressor
                best_prediction = prediction
        best_errors.append(best_error)
        best_models.append(best_model)
        best_predictions.append(best_prediction)
            
    #print(best_models)
    #print(best_errors)
    #print(best_predictions)
    #time.sleep(5)

    """
    #print("prediction list: ", prediction_list)

    for i in range(0, len(y_test)):
        best_error = sys.maxsize
        for key_, value in errors_list.items():
            print("Model: ", key_, " Error: ", value[i])
            if value[i] < best_error:
            #print("Model: ", key_, " Error: ", value[i])
            #if "SVR()" in str(key_):
                best_error = value[i]
                best_model = key_
        print("Best model: ", best_model, " Best error: ", best_error)
        time.sleep(2)
        best_models.append(best_model)
        best_errors.append(best_error)
        best_predictions.append(prediction_list[best_model][i])

    print(best_models)
    #print(best_errors)
    #print(best_predictions)

    final_elem = {}
    for i in range(1, PREDICTION_WINDOW+1, PREDICTION_WINDOW_step):
        final_elem[i] = []

    for l in best_predictions:
        i = 1
        for v in l:
            final_elem[i].append(v)
            i = i + PREDICTION_WINDOW_step
    """

    for key_, value in best_prediction.items():
        print("Key: ", key_, " Value in final elem: ", value, " y_test: ", y_test[-len(value):])
        #time.sleep(2)
        mse = mean_squared_error(y_test[-len(value):], value)
        mae = mean_absolute_error(y_test[-len(value):], value)
        result[key_] = {"mse": mse, "mae": mae}

    #print("Result: ", result)
    #print(result)
    res_final[key] = best_prediction

print(res_final)
#sys.exit(0)
# Save the results to a file
with open('../data/adaptive-model-continuous-24h-predictions-6.json', 'w') as file:
    json.dump(res_final, file)