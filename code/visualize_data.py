import sys
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import random
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from pandas import DataFrame
import time
import warnings

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    
    # make predictions
    predictions = []
    
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

def evaluate_models(dataset, p_values, d_values, q_values):
    #dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return best_cfg

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

log_filename_expe = "../data/merged-data.rawdata"
log_filename_simu = "../data/cooja-data.rawdata"

files = [log_filename_expe]
series_list = [] # List containing both data from expe and simulation for comparison

for log_filename in files:
    try:
        log_file = open(log_filename, "r" )
    except IOError:
        print(sys.argv[0]+": "+log_filename+": cannot open file")
        sys.exit(3)

    data = log_file.readlines()

    communication_dict = {}

    list_nodes = []

    for line in data:
        receiver = "Received" in line
        sender = "Sending" in line
        if not sender and not receiver :
            continue
        #print(line.replace(';',' ').split())
        timestamp, node_id, event_type, message_id = line.replace(';',' ').split()

        if node_id not in list_nodes:
            list_nodes.append(node_id)

        if event_type == "SendingBroadcast":
            sender = node_id
            message_id = message_id.strip() 
            if message_id not in communication_dict:
                communication_dict[message_id] = {"sender": sender, "receivers_list": [], "timestamps": []}

        elif event_type == "Received":
            receiver = node_id
            message_id = message_id.strip()  # Remove leading space

            #print(communication_dict)
            if message_id in communication_dict:
                communication_dict[message_id]["receivers_list"].append(receiver)
                communication_dict[message_id]["timestamps"].append(str(timestamp))
                if sender == receiver:
                    print("Sender = Receiver ", message_id)
                    time.sleep(5)
            else:
                print("Message ", message_id," received before sending")

    # Print the resulting dictionary
    print(communication_dict)

    temp_dict = {}
    for key, value in communication_dict.items():
        sender = value.get("sender")
        if sender not in temp_dict:
            temp_dict[sender] = []
            
        i = 0
        l = []
        for elem in value.get("receivers_list"):
            elem = elem + ":"+str(value.get("timestamps")[i])
            i = i + 1
            l.append(elem)
        temp_dict[sender].append(l)
        #temp_dict[sender].append(value.get("receivers_list"))
        #temp_dict[sender].append(value.get("timestamps"))

    print(temp_dict)
    #time.sleep(10)
    first_node = 95 # To change here according to the data coming from expe or simu
    #first_node = 1
    nb_nodes = 5

    series_dict = {}

    #print(list_nodes)
    # Changed all node ids from ID to m3
    for key, value in temp_dict.items():
        print("yes ", key)
        time.sleep(5)
        sender = key
        cpt = 1

        # Loop over all the other nodes
        node = first_node
        node_ids = []
        while cpt <= nb_nodes:
            if ("m3-"+str(node) in list_nodes):
                node_ids.append(node)
                receiver = "m3-"+str(node)
                new_key = sender+'_'+receiver
                series_dict[new_key] = []
                for list in value:
                    if receiver in list:
                        series_dict[new_key].append("1:"+timestamp)
                    else:
                        series_dict[new_key].append(0)
                cpt = cpt + 1
                node = node + 1
            else:
                node = node + 1

    print(series_dict)
    series_list.append(series_dict)

    """
    sender = "m3-110"
    receiver = "m3-111"

    stop = False
    while stop == False:
        random_senders = random.sample(node_ids, 3)
        random_receivers = random.sample(node_ids, 3)
        if not any(element in random_senders for element in random_receivers):        
            stop = True
    """

random_senders = random.sample(node_ids, 3)
random_receivers = random.sample(node_ids, 3)
Position = range(1,10)

k = -1

data_expe = series_list[0]
data_simu = series_list[1]

for n in random_senders:
    for m in random_receivers:
        k = k + 1
        print(k)
        sender = "m3-"+str(n)
        receiver = "m3-"+str(m)

        for elem in series_dict:
            print(elem)
        key = sender+"_"+receiver

        data_expe_values = data_expe[key]
        data_simu_values = data_simu[key]

        # Calculating PDR series
        pdr_data_expe = []
        pdr_data_simu = []

        i = 0
        window_size = 40 # Window on transmissions

        while i < len(data_expe_values):
            try:
                sum = 0
                for j in range(0, window_size):
                    sum = sum + data_expe_values[i+j]
                avg = sum / window_size
                pdr_data_expe.append(avg)
                i = i + window_size
            except:
                print("Exception occured")
                break

        print(pdr_data_expe, len(pdr_data_expe))

        while i < len(data_simu_values):
            try:
                sum = 0
                for j in range(0, window_size):
                    sum = sum + data_simu_values[i+j]
                avg = sum / window_size
                pdr_data_simu.append(avg)
                i = i + window_size
            except:
                print("Exception occured")
                break

        print(pdr_data_expe, len(pdr_data_expe))
        print(pdr_data_simu, len(pdr_data_simu))

        time.sleep(3)

        x = np.arange(1, len(pdr_data_expe) + 1) 
        y_expe = np.array(pdr_data_expe)
        y_simu = np.array(pdr_data_simu)
        
        fig = plt.figure(1)
        fig.tight_layout(pad=0.5)

        print (k, Position[k])
        ax = fig.add_subplot(3,3,Position[k])
        ax.set_title(key)
        ax.plot(y_expe, label='expe')
        ax.plot(y_simu, label='simu')

        #plt.show()
        # plotting
        #plt.title(key) 
        #plt.xlabel("X axis") 
        #plt.ylabel("Y axis") 
        #plt.plot(x, y, color ="blue") 
        #plt.show()
plt.show()

k = -1

for n in random_senders:
    for m in random_receivers:
        k = k + 1
        print(k)
        sender = "m3-"+str(n)
        receiver = "m3-"+str(m)

        for elem in series_dict:
            print(elem)
        key = sender+"_"+receiver
        data = series_dict[key]

        # Calculating PDR series
        pdr_data = []

        i = 0
        window_size = 40

        while i < len(data):
            try:
                sum = 0
                for j in range(0, window_size):
                    sum = sum + data[i+j]
                avg = sum / window_size
                pdr_data.append(avg)
                i = i + window_size
            except:
                print("Exception occured")
                break

        print(pdr_data)

        data = pdr_data

        #x = np.arange(1, len(data) + 1) 
        #y = np.array(data)
        p_values = [4, 5, 6]
        d_values = range(2, 4)
        q_values = range(2, 4)
        warnings.filterwarnings("ignore")
        #best_cfg = evaluate_models(data, p_values, d_values, q_values)
        best_cfg = (5,4,3)
        
        # split into train and test sets
        X = data
        size = int(len(X) * 0.66) # First 2/3 of the data are used for training, and the rest is used for testing
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = []
        # walk-forward validation
        #print(train, test, size)

        cpt = 0
        for t in range(len(test)):
            model = ARIMA(history, order=best_cfg)
            model_fit = model.fit()
            output = model_fit.forecast()
            cpt = cpt + 1
            if (cpt > 1): # Ignore the first prediction
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
                print('predicted=%f, expected=%f' % (yhat, obs))

        # evaluate forecasts
        #rmse = sqrt(mean_squared_error(test, predictions))
        #print('Test RMSE: %.3f' % rmse)
        # plot forecasts against actual outcomes
        fig = plt.figure(1)
        fig.tight_layout(pad=0.5)

        print (k, Position[k])
        ax = fig.add_subplot(3,3,Position[k])
        ax.set_title(key)
        ax.plot(test, label='observations')      # Or whatever you want in the subplot
        ax.plot(predictions, color='red', label='predictions')
        plt.legend()
        
plt.show()