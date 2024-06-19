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

SLA = 40 # Required number of packets correctly received in a window of 50 seconds

def evaluate_model(model, x, y):
    # Evaluate the model
    y_pred = model.predict(x)

    if 'Sequential' in str(regressor): # LSTM
        y_pred = scaler.inverse_transform(y_pred)
        #test_predict = scaler.inverse_transform(test_predict)

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(y_test)):
        if y[i] <= SLA and y_pred[i] <= SLA:
            true_positive += 1
        elif y[i] >= SLA and y_pred[i] >= SLA:
            true_negative += 1
        elif y[i] > SLA and y_pred[i] < SLA:
            false_positive += 1
            #print("here, ", i, y[i], y_pred[i])
            #time.sleep(1)
        elif y[i] < SLA and y_pred[i] > SLA:
            false_negative += 1
            #print("here, ", i, y[i], y_pred[i])
            #time.sleep(1)

    y_true = np.array([1 if val >= SLA else 0 for val in y])
    y = y_true[:len(y_pred)]
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    # Applying softmax along axis 1
    probabilities = np.array(softmax(y_pred))
    #print(probabilities)

    #time.sleep(1)

    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y, probabilities)
    roc_auc = auc(fpr, tpr)
    return accuracy, roc_auc, fpr, tpr, thresholds


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
    
    """
    fig = plt.figure(1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    # Initialize lists to hold the legend handles and labels
    handles = []
    labels = []

    
    # Loop over your subplots and collect handles and labels
    ax = fig.add_subplot(4, 4, Position[k])  # Adjust according to the number of subplots
    ax.set_title(key)  # Replace with your actual title or key
    line1, = ax.plot(y_exp, label='experiments') 
    line2, = ax.plot(y_sim, label='simulation')  
    line3, = ax.plot(y_sim_pdr, label='simulation_pdr')

    # Collect the handles and labels
    if k == 0:  # Collect only once, to avoid duplicates
        handles.extend([line1, line2, line3])
        labels.extend([line1.get_label(), line2.get_label(), line3.get_label()])

    fig.legend(handles, labels, loc='upper center')
    """

#plt.show()

k = -1
mae_list = []
mse_list = []
r_squared_list = []
rmse_list = []

steps_list = [19, 21, 23, 25]
elems = []
for steps in steps_list:
    results = []
    elem = {}

    #print("Steps: ", steps)

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

        randomForestRegressor = RandomForestRegressor(n_estimators = 100, random_state = 1)
        linearRegressor = LinearRegression()
        svr = SVR(kernel='rbf')
        gradientBoostingRegressor = GradientBoostingRegressor()
        decisionTreeRegressor = DecisionTreeRegressor()
        adaBoostRegressor = AdaBoostRegressor()
        extraTreesRegressor = ExtraTreesRegressor()
        kNeighborsRegressor = KNeighborsRegressor()
        lstm = Sequential()

        #regressors = [randomForestRegressor, linearRegressor, svr, gradientBoostingRegressor, decisionTreeRegressor, adaBoostRegressor, extraTreesRegressor, kNeighborsRegressor]
        regressors = [lstm]
        max_accuracy = 0
        min_auc = 0

        for regressor in regressors:
            time_steps = [3, 5, 10, 15, 20]
            for time_step in time_steps:
                # Prepare sequences
                def create_sequences(data, time_step):
                    X, y = [], []
                    for i in range(len(data) - time_step):
                        a = data[i:(i + time_step)]
                        X.append(a)
                        y.append(data[i + time_step])
                    #print(data, X, y)
                    #print(X[-1], y[-1])
                    #time.sleep(5)
                    return np.array(X), np.array(y)

                X_train, y_train = create_sequences(train, time_step)
                X_test, y_test = create_sequences(test, time_step)
                
                #print(str(regressor))
                # Compile the model
                
                if 'Sequential' in str(regressor): # LSTM
                    regressor = Sequential()
                    # Normalize data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_train = scaler.fit_transform(np.array(train).reshape(-1, 1))
                    scaled_test = scaler.transform(np.array(test).reshape(-1, 1))

                    X_train, y_train = create_sequences(scaled_train, time_step)
                    X_test, y_test = create_sequences(scaled_test, time_step)
                    
                    # Reshape input to be [samples, time steps, features]
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    regressor.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
                    regressor.add(LSTM(50, return_sequences=False))
                    regressor.add(Dense(25))
                    regressor.add(Dense(1))
                    regressor.compile(optimizer='adam', loss='mean_squared_error')
                    
                regressor.fit(X_train, y_train)
                accuracy, roc_auc, fpr, tpr, thresholds = evaluate_model(regressor, X_train, y_train)
                #print(regressor, accuracy, roc_auc)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    #min_auc = roc_auc
                    best_regressor = regressor
                    best_time_step = time_step
                    """
                    # Make predictions
                    train_predict = regressor.predict(X_train)
                    test_predict = regressor.predict(X_test)

                    # Plot the results
                    train_plot = np.empty_like(X, dtype=float)
                    train_plot[:] = np.nan

                    test_plot = np.empty_like(X, dtype=float)
                    test_plot[:] = np.nan
                    if 'Sequential' in str(regressor): # LSTM
                        train_predict = scaler.inverse_transform(train_predict)
                        test_predict = scaler.inverse_transform(test_predict)

                        train_plot[time_step:len(train_predict) + time_step] = train_predict[:, 0]
                        test_plot[len(train_predict) + (time_step * 2):len(X)] = test_predict[:, 0]
                    else:
                        train_plot[time_step:len(train_predict) + time_step] = train_predict
                        test_plot[len(train_predict) + (time_step * 2):len(X)] = test_predict
                    """

        #accuracy, roc_auc, fpr, tpr, thresholds = evaluate_model(best_regressor, X_test, y_test)
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        index = 0
        #print("Best regressor: ", best_regressor, " with accuracy (train): ", accuracy, " and learning step: ", best_time_step)
        
        for _ in range(len(test)//steps):
            try:
                model = best_regressor
                #model.initialize_approximate_diffuse() # this line is added to avoir the LU decomposition error
                X_train, y_train = create_sequences(history, best_time_step)
                X_test, y_test = create_sequences(test, best_time_step)
                
                model_fit = model.fit(X_train, y_train)
                if 'Sequential' in str(regressor): # LSTM
                    forecast = model.predict(X_test)[:steps]
                    #print(forecast)
                else:
                    forecast = model_fit.predict(X_test)[:steps]

                #print("forecast: ", forecast)
                #output = model_fit.predict(len(history), len(history)+steps, dynamic=True)
                #time.sleep(2)

                for t in range(index, index + steps):
                    obs = test[t]
                    history.append(obs)
                    yhat = forecast[t-index]

                    if yhat < 0:
                        yhat = 0
                    elif yhat > 50:
                        yhat = 50

                    if yhat <= SLA and obs <= SLA:
                        true_positive = true_positive + 1
                    elif yhat >= SLA and obs <= SLA:
                        false_negative = false_negative + 1
                    elif yhat <= SLA and obs >= SLA:
                        false_positive = false_positive + 1
                    elif yhat >= SLA and obs >= SLA:
                        true_negative = true_negative + 1
                    if 'Sequential' in str(regressor): # LSTM
                        predictions.append(str(yhat[0]))
                    else:
                        predictions.append(yhat)
                    #print(key, "predicted=%f, expected=%f" % (yhat, obs))

                index = index + steps
    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print("Exception occured", e)
                pass

        result[key] = {"regressor": str(best_regressor), "prediction": predictions, "true_positive": true_positive, "false_positive": false_positive, "true_negative": true_negative, "false_negative": false_negative}
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        #print("Accuracy (test): ", accuracy)
        results.append(result)           
        
        """
        #time.sleep(1)
        regressor = best_regressor
        #print(X, train_plot)
        plt.plot(X, label='Original Data')
        plt.plot(train_plot, label='Training Prediction')
        plt.plot(test_plot, label='Testing Prediction of model: '+str(regressor))
        plt.legend()
        plt.show()
        """
            
        #print("False Positive: ", false_positive, " False Negative: ", false_negative, " True Positive: ", true_positive, " True Negative: ", true_negative)

        # Print the evaluation metrics
        #print("Mean Absolute Error (MAE):", mae)
        #print("Mean Squared Error (MSE):", mse)
        #print("R-squared (R²):", r_squared)
        #print("Root Mean Squared Error (RMSE):", rmse)

        """
        fig = plt.figure(1)
        #fig.tight_layout(pad=0.5)
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        #print (k, Position[k])
        ax = fig.add_subplot(4,4,Position[k])
        ax.set_title(key)

        size_simu = int(len(data_simu_values) * 0.66) # First 2/3 of the data are used for training, and the rest is used for testing
        train_simu, test_simu = data_simu_values[0:size_simu], data_simu_values[size_simu:len(data_simu_values)] 
        train_simu_pdr, test_simu_pdr = data_simu_pdr_values[0:size_simu], data_simu_pdr_values[size_simu:len(data_simu_pdr_values)]

        # Initialize lists to hold the legend handles and labels
        handles = []
        labels = []

        # Loop over your subplots and collect handles and labels
        line1, = ax.plot(test, label='experiments') 
        line2, = ax.plot(test_simu, label='simulation')  
        line3, = ax.plot(test_simu_pdr, label='simulation_pdr')
        line4, = ax.plot(predictions, label='predictions')

        # Collect the handles and labels
        if k == 0:  # Collect only once, to avoid duplicates
            handles.extend([line1, line2, line3, line4])
            labels.extend([line1.get_label(), line2.get_label(), line3.get_label(), line4.get_label()])

        fig.legend(handles, labels, loc='upper center')
        """
    elem [steps] = {"results": results, "time_step": best_time_step}
    elems.append(elem)    
            
    #print('Mean Absolute Error (MAE): ', mae_list, np.mean(mae_list))
    #print('Mean Squared Error (MSE): ', mse_list, np.mean(mse_list))
    #print('R-squared (R²): ', r_squared_list, np.mean(r_squared_list))
    #print('Root Mean Squared Error (RMSE): ', rmse_list, np.mean(rmse_list))
    #print("True Positive: ", true_positive, " False Positive: ", false_positive, " True Negative: ", true_negative, " False Negative: ", false_negative)
    #print("====================================")
    #plt.title("Step: "+str(steps))
    #plt.show()

filename = sys.argv[1].split(".")[0]
#print(elems)
#print(filename)
with open('../data/'+filename+'.json', 'a') as file:
    json.dump(elems, file) 