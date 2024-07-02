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

def evaluate_model(model, x, y):
    # Evaluate the model
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    if 'Sequential' in str(regressor): # LSTM
        y_pred = scaler.inverse_transform(y_pred)
        #test_predict = scaler.inverse_transform(test_predict)

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    
    return mse, mae


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
#couples = [(2, 10), (2, 9)]
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

    #print(key, " with data expe size: ", len(data_expe_values))
    #print(key, " with data simu size: ", len(data_simu_values))

    data_simu_values = data_simu_values[0:len(data_expe_values)] # Cut the simulation data to match the size of the experiments data
    data_simu_pdr_values = data_simu_pdr_values[0:len(data_expe_values)] # Cut the simulation data to match the size of the experiments data

    x = np.arange(1, len(data_simu_values) + 1) 
    y_exp = np.array(data_expe_values)
    y_sim = np.array(data_simu_values)
    y_sim_pdr = np.array(data_simu_pdr_values)

#plt.show()

k = -1
mae_list = []
mse_list = []
r_squared_list = []
rmse_list = []

steps_list = [1, 2, 3, 5, 7, 10, 15, 20]
elems = []
for prediction_step in steps_list:
    results = []
    elem = {}

    for n, m in couples:
        result = {}
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

        #print(X)

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
        #regressors = [kNeighborsRegressor] # Seems like that there is a problem with KNeighborsRegressor
        regressor_strings = [str(regressor) for regressor in regressors]

        #regressors = [lstm]
        max_accuracy = 0
        min_auc = 0
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
                    #print(data, X, y)
                    #print(X[-1], y[-1])
                    #time.sleep(5)
                    return np.array(X), np.array(y)

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

                elif "randomForestRegressor" in regressor_string:
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

                mse, mae = evaluate_model(regressor, X_train, y_train)

                #print(regressor, mse, mae)
                #time.sleep(1)
                if mse < min_mse:
                    min_mae = mae
                    min_mse = mse
                    best_regressor = regressor
                    best_window_step = window_step

        print("couple: ", (n,m), " step: ", prediction_step, " Best regressor: ", best_regressor, " window step: ", best_window_step, " MAE: ", min_mae, " MSE: ", min_mse)
    
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
        elif "randomForestRegressor" in regressor_string:
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
        #print("new test: ", test)
        #time.sleep(2)
        new_test = [v for v in test]
        inputs = []
        index = 0
        predictions = []

        while index < len(test):
            new_test = [v for v in test]
            try:
                for i in range(index, index+prediction_step):
                    input_window = new_test[i:window_step+i]
                    #print("input window: ", input_window)
                    #time.sleep(1)
                    reshaped_input_window = np.array(input_window).reshape(1, -1)
                    #print("reshaped input window: ", reshaped_input_window)
                    inputs.append(reshaped_input_window)
                    # Predict the next value
                    predicted_value = regressor.predict(reshaped_input_window)
                    #print("predicted value: ", predicted_value)
                    predictions.append(predicted_value[0])
                        
                    # Replace the element in the TestSet with the predicted value
                    new_test[window_step+i] = predicted_value[0]
                    i = i + 1
                    #time.sleep(2)

                #print("test: ", test, index, window_step)
                refit_data = test[index:index+window_step+prediction_step]
                #print("refit data: ", refit_data)
                            
                # Prepare data for refitting
                X_train, y_train = create_sequences(refit_data, window_step)
                #print("Refit with: X train: ", X_train, "y train: ", y_train)
                #time.sleep(1)
                index = index + prediction_step
                        
                # Refit the model
                regressor.fit(X_train, y_train)
    
            except KeyboardInterrupt:
                break
            except Exception as e:
                #print("Exception occured", e, " Index: ", index, len(test))
                break

        #print(test[-len(predictions):])
        #print(predictions)
        mse = mean_squared_error(test[-len(predictions):], predictions)
        mae = mean_absolute_error(test[-len(predictions):], predictions)
        result[key] = {"regressor": str(best_regressor), "window_step": best_window_step, "MAE": mae, "MSE": mse, "predictions": predictions}
        print("prediction step: ", prediction_step, ", mse: ", mse, ", mae: ", mae)
        mae_list.append(mae)
        mse_list.append(mse)
        #time.sleep(1)
        results.append(result)
        
    elem [prediction_step] = {"results": results, "window_step": best_window_step}
    elems.append(elem)    
            
    print('Mean Absolute Error (MAE) for step: ', str(prediction_step), ": ", np.mean(mae_list))
    print('Mean Squared Error (MSE): ', np.mean(mse_list))

filename = sys.argv[1].split(".")[0]
print(elems)
#print(filename)
with open('../data/'+filename+'.json', 'a') as file:
    json.dump(elems, file)