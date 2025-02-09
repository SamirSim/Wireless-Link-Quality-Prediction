from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mango import Tuner
import matplotlib.pyplot as plt # type: ignore
import json
import numpy as np
import warnings
import random
import time
import sys
import pmdarima as pm

def arima_objective_function(args_list):
    global data_values
    
    params_evaluated = []
    results = []
    
    for params in args_list:
        try:
            p,d,q = params['p'],params['d'], params['q']
            trend = params['trend']
            
            model = ARIMA(data_values, order=(p,d,q), trend = trend)
            predictions = model.fit()
            mse = mean_squared_error(data_values, predictions.fittedvalues)   
            params_evaluated.append(params)
            results.append(mse)
        except:
            #print(f"Exception raised for {params}")
            #pass 
            params_evaluated.append(params)
            results.append(1e5)
        
        #print(params_evaluated, mse)
    return params_evaluated, results

param_space = dict(p= range(0, 30),
                   d= range(0, 30),
                   q =range(0, 30),
                   trend = ['n', 'c', 't', 'ct']
                  )


################# Main #################
log_filename_expe = "../data/expe-data-grenoble.rawdata"
#log_filename_simu = "../data/cooja-grenoble-p1-reduced.rawdata"
#log_filename_simu = "../data/cooja-grenoble-p0,56.rawdata"
log_filename_simu = "../data/cooja-grenoble-customized-p.rawdata"

already_executed = True

period = 50 # in seconds

if already_executed:
    # Load the series_list from the file
    with open('../data/series_list_customized_p.json', 'r') as file:
        series_list = json.load(file)

    data_expe = series_list[0]
    data_simu = series_list[1]

    with open('../data/series_list_p0,56.json', 'r') as file:
        series_list = json.load(file)

    data_simu_pdr = series_list[1]

couples = [(2, 10), (2, 9), (7, 9), (7, 6), (10, 6), (10, 2), (11, 2), (11,6), (4, 5), (4, 6), (5, 6), (5, 4), (6, 4), (6, 5), (6, 10), (6, 9)]
#couples = [(5, 6)]
Position = range(1,17)

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

plt.show()

k = -1
mae_list = []
mse_list = []
r_squared_list = []
rmse_list = []

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

    warnings.filterwarnings("ignore")
        
    # split into train and test sets
    X = data_expe_values
    size = int(len(X) * 0.66) # First 2/3 of the data are used for training, and the rest is used for testing
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = []

    # Define the ranges for p, d, and q
    start_p = 2
    start_d = 2
    start_q = 2

    max_p = 7
    max_d = 7
    max_q = 7

    # Initialize variables to store the best model and lowest MSE
    best_model = None
    lowest_mae = float('inf')
    lowest_mse = float('inf')
    lowest_score = float('inf')

    index = 0
    steps = 1
    data_values = history

    for p in range(start_p, max_p + 1):
        for d in range(start_d, max_d + 1):
            for q in range(start_q, max_q + 1):
                model = ARIMA(history, order=(p,d,q))
                model.initialize_approximate_diffuse()

                pred = model.fit().predict(start=0, end=len(history)-1, dynamic=False)
            
                mae = mean_absolute_error(history, pred)
                mse = mean_squared_error(history, pred)
                rmse = np.sqrt(mse)

                if np.std(pred) < 0.01: # Penalize the model if the standard deviation of the predictions is too low
                    mse *= 2
                #mape = np.mean(np.abs((history - pred) / history)) * 100

                # Normalize the evaluation metrics
                mae_normalized = mae / (max(history) - min(history))
                mse_normalized = mse / ((max(history) - min(history)) ** 2)
                #mape_normalized = mape / 100

                # Calculate the score as the sum of normalized metrics
                score = mae_normalized + mse_normalized # + mape_normalized

                #print(p, d, q, mae, mse, rmse, score)

                if score < lowest_score:
                    lowest_score = score
                    best_model = model

                #print (p, d, q, mae, mse, rmse, mape)

    # Get the order of the model
    p, d, q = best_model.order
    best_cfg = (p, d, q)

    # Print the order
    print(f"The optimal (p, d, q) is: {p, d, q}, for ", key, " and the score is: ", lowest_score)
    #time.sleep(2)

    for _ in range(len(test)//steps):
        try:
            model = ARIMA(data_values, order=best_cfg)
            model.initialize_approximate_diffuse() # this line is added to avoir the LU decomposition error
            model_fit = model.fit()
            forecast = model_fit.forecast(steps)

            for t in range(index, index + steps):
                obs = test[t]
                data_values.append(obs)
                yhat = forecast[t-index]
                predictions.append(yhat)
                #print('predicted=%f, expected=%f' % (yhat, obs))
                #time.sleep(1)
            index = index + steps
 
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Exception occured", e)
            pass
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(test[:len(predictions)], predictions)
    mse = mean_squared_error(test[:len(predictions)], predictions)
    r_squared = r2_score(test[:len(predictions)], predictions)
    rmse = np.sqrt(mse)

    # Print the evaluation metrics
    print("Mean Squared Error (MSE): for (p, d, q)", mse, best_cfg)
    #print("Mean Squared Error (MSE):", mse)
    #print("R-squared (R²):", r_squared)
    #print("Root Mean Squared Error (RMSE):", rmse)

    mae_list.append(mae)
    mse_list.append(mse)
    r_squared_list.append(r_squared)
    rmse_list.append(rmse)

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
    #line5, = ax.plot(history, label='history')

    # Collect the handles and labels
    if k == 0:  # Collect only once, to avoid duplicates
        handles.extend([line1, line2, line3, line4])
        labels.extend([line1.get_label(), line2.get_label(), line3.get_label(), line4.get_label()])

    fig.legend(handles, labels, loc='upper center')
        
print('Step: ', steps)
print('Mean Absolute Error (MAE): ', mae_list, np.mean(mae_list))
print('Mean Squared Error (MSE): ', mse_list, np.mean(mse_list))
print('R-squared (R²): ', r_squared_list, np.mean(r_squared_list))
print('Root Mean Squared Error (RMSE): ', rmse_list, np.mean(rmse_list))
print("====================================")

plt.show()