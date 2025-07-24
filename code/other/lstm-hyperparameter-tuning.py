import sys
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from pandas import read_csv # type: ignore
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from kerastuner.tuners import RandomSearch
from statsmodels.tsa.arima.model import ARIMA # type: ignore
from math import sqrt
from pandas import DataFrame # type: ignore
import time
import warnings
import random
import string
import json

random.seed(10)

SLA = 40 # Required number of packets correctly received in a window of 50 seconds

def replace_message_id_in_remaining_lines(file_path, message_id, new_message_id):
    # Used to replace the identical msg ids for two different nodes
    updated_lines = []

    with open(file_path, 'r') as file:
        cpt = 0
        line = 1
        while line:
            line = file.readline()
            words = line.split(';')
            if "SendingBroadcast" in line and message_id == words[-1]:
                cpt = cpt + 1
                if cpt == 2: # Look for the second occurence of the msg_id
                    break
            updated_lines.append(line)
        
        #print(line)
        #time.sleep(1)
        while line: 
            if message_id not in line:
                updated_lines.append(line)  # Append remaining lines as they are
            else:                    
                new_line = line.replace(message_id, new_message_id + '\n')
                updated_lines.append(new_line)
                #print("updated here: ", new_line)
            line = file.readline()

    #print("updated lines: ", updated_lines)
    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

    return updated_lines

def generate_random_string(length):
    # Used to replace the identical ids among different nodes in Cooja
    # Generate a random string of length "length"
    characters = string.ascii_letters + string.digits  # You can add more characters if needed

    # Generate the random string
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string

def convert_time_to_seconds_milliseconds(time_str):
    # Used for translating the timing format from Cooja's to FIT IoT-Lab's 
    # Split the time string into hours, minutes, seconds, and milliseconds
    parts = time_str.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        print(len(parts), parts)
        raise ValueError("Invalid time format")

    # Convert to integers
    hours = int(hours)
    minutes = int(minutes)
    seconds = float(seconds)

    # Convert hours to seconds
    total_seconds = hours * 3600

    # Add minutes to total seconds
    total_seconds += minutes * 60

    # Add remaining seconds
    total_seconds += seconds

    return total_seconds

def evaluate_arima_model(X, arima_order):
    # Evaluate an ARIMA model for a given order (p,d,q)
    
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    
    # make predictions
    pred = []
    
    model = ARIMA(history, order=arima_order)
    model.initialize_approximate_diffuse()
    pred = model.fit().predict(start=0, end=len(history)-1, dynamic=False)
            
    mae = mean_absolute_error(history, pred)
    mse = mean_squared_error(history, pred)

    if np.std(pred) < 0.01: # Penalize the model if the standard deviation of the predictions is too low
        mse *= 2

    # Normalize the evaluation metrics
    mae_normalized = mae / (max(history) - min(history))
    mse_normalized = mse / ((max(history) - min(history)) ** 2)
    #mape_normalized = mape / 100

    # Calculate the score as the sum of normalized metrics
    score = mae_normalized + mse_normalized # + mape_normalized

    #print(arima_order, score)
    return score

def optimize_model(dataset, p_values, d_values, q_values):
    # Find the best configuration (p,d,q) for ARIMA model using grid search
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    score = evaluate_arima_model(dataset, order)
                    #print("here, ", score, best_score)
                    if score < best_score:
                        best_score, best_cfg = score, order
                    #print('ARIMA%s MSE=%.3f' % (order,score))
                except:
                    continue
    #print('Best ARIMA%s Score=%.3f' % (best_cfg, best_score))
    return best_cfg

def get_mean_pdr(data):
    i = 0
    sum = 0

    while i < len(data):
        try:
            sum = sum + data[i]
            i = i + 1
        except:
            print("Exception occured")
            break
    avg = sum / len(data)

    return avg

################# Main #################
log_filename_expe = "../data/expe-data-grenoble.rawdata"
#log_filename_simu = "../data/cooja-grenoble-p1-reduced.rawdata"
#log_filename_simu = "../data/cooja-grenoble-p0,56.rawdata"
log_filename_simu = "../data/cooja-grenoble-customized-p.rawdata"

already_executed = True

period = 50 # in seconds
if not already_executed:
    files = [log_filename_expe, log_filename_simu]
    series_list = [] # List containing both data from expe and simulation for comparison

    found = 0 # Used to determine if two identical message ids have been found for two different nodes
    for log_filename in files:
        try:
            log_file = open(log_filename, "r" )
        except IOError:
            print(sys.argv[0]+": "+log_filename+": cannot open file")
            sys.exit(3)

        data = log_file.readlines()

        communication_dict = {}
        list_nodes = []
        message_ids = []
        new_data = []

        for line in data:
            receiver = "Received" in line
            sender = "SendingBroadcast" in line
            if not sender and not receiver :
                continue
            
            try:
                timestamp, node_id, event_type, message_id = line.split(";")
            except:
                print(line)
                break

            if "cooja" in log_filename:
                #print("here")
                timestamp = convert_time_to_seconds_milliseconds(timestamp)

            if node_id not in list_nodes:
                list_nodes.append(node_id)

            if event_type == "SendingBroadcast":
                if message_id in message_ids: # Found two identical message ids in different nodes
                    new_message_id = generate_random_string(10)
                    data = replace_message_id_in_remaining_lines(log_filename, message_id, new_message_id)
                    #data = [line.replace(message_id, new_message_id+'\n') for line in data]
                    print ("replaced ", message_id, " with ", new_message_id)
                    found = 1
                    message_id = new_message_id
                
                message_ids.append(message_id)
                sender = node_id
                message_id = message_id.strip() 
                if message_id not in communication_dict:
                    communication_dict[message_id] = {"sender": sender, "receivers_list": [], "timestamp": timestamp}

            elif event_type == "Received":
                receiver = node_id
                message_id = message_id.strip()  # Remove leading space
    
                if message_id in communication_dict:
                    communication_dict[message_id]["receivers_list"].append(receiver)
                else:
                    print("Message ", message_id," received before sending")

        time.sleep(3)

        # Print the resulting dictionary
        print(communication_dict)

        temp_dict = {}

        for key, value in communication_dict.items():
            sender = value.get("sender")
            # Add timestamp in the beginning of the receivers_list
            l = [value.get("timestamp"), value.get("receivers_list")]
            if sender not in temp_dict:
                temp_dict[sender] = []
            temp_dict[sender].append(l)
        print("=====================")
        print("temp dict: ", temp_dict)
        time.sleep(2)
        
        series_dict = {}
        first_timestamp = 0

        for key, value in temp_dict.items():
            first_timestamp = value[0][0]
            l = []
            save = []
            for list in value:
                if float(list[0]) >= float(first_timestamp) + float(period):
                    save.append(l)
                    l = []
                    first_timestamp = list[0]
                
                for elem in list[1]:
                    l.append(elem)
            series_dict[key] = save
        
        print("=====================")      
        # Structure of series_dict: For each node, there is a list of the reached nodes during a window with length 'period', successively
        print("series_dict: ", series_dict)
        time.sleep(2)  

        first_node = 2
        nb_nodes = 11
        final_dict = {}
        for key, value in series_dict.items():
            sender = key
            cpt = 1

            # Loop over all the other nodes
            node = first_node
            node_ids = []
            try:
                first_timestamp = value[0][0]
            except:
                print(key, value)
                time.sleep(2)
            
            l = []
            while cpt <= nb_nodes:
                if "m3-"+str(node) in list_nodes:
                    l = []
                    node_ids.append(node)
                    receiver = "m3-"+str(node)
                    new_key = sender+'_'+receiver
                    final_dict[new_key] = []

                    for list in value:
                        if receiver in list:
                            final_dict[new_key].append(list.count(receiver))
                        else:
                            final_dict[new_key].append(0)
                    cpt = cpt + 1
                    node = node + 1
                else:
                    node = node + 1

        print("=====================")
        print(final_dict)
        time.sleep(2)

        series_list.append(final_dict)
    
    # Store series_list in a file in JSON format
    with open('../data/series_list_customized_p.json', 'w') as file:
        json.dump(series_list, file)

    random_senders = random.sample(node_ids, 3)
    random_receivers = random.sample(node_ids, 3)
    Position = range(1,10)

    k = -1

    data_expe = series_list[0]
    data_simu = series_list[1]
else:
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

couples = [(2, 10), (2, 9), (7, 9), (7, 6), (10, 6), (10, 2), (11, 2), (11,6), (4, 5), (4, 6), (5, 6), (5, 4), (6, 4), (6, 5), (6, 10), (6, 9)]
#couples = [(2, 10), (2, 9)]
k = -1
mean_pdr_total = 0

for key, value in data_expe.items():
    mean_pdr = get_mean_pdr(value) / period # This only works because the sending period is the same for all nodes and equal to 1
    #print(key, mean_pdr, len(value))
    mean_pdr_total = mean_pdr_total + mean_pdr
#print("Mean PDR of the network: ", mean_pdr_total/len(data_expe))

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

steps_list = [1, 5]
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

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(np.array(train).reshape(-1, 1))
        scaled_test = scaler.transform(np.array(test).reshape(-1, 1))

        # Prepare sequences
        def create_sequences(data, time_step):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                a = data[i:(i + time_step), 0]
                X.append(a)
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        time_step = 1  # Number of past observations to use
        X_train, y_train = create_sequences(scaled_train, time_step)
        X_test, y_test = create_sequences(scaled_test, time_step)

        # Define a function to build the LSTM model
        def build_model(hp):
            model = Sequential()
            model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                        return_sequences=True,
                        input_shape=(time_step, 1)))
            model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
            model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32)))
            model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
            model.add(Dense(1))
            
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                        loss='mse')
            return model

        # Define the tuner
        tuner = RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=20,
            executions_per_trial=1,
            directory='my_dir',
            project_name='lstm_tuning')

        # Metrics
        false_negative = 0
        false_positive = 0
        true_positive = 0
        true_negative = 0 

        # Make predictions
        #train_predict = model.predict(X_train)
        #test_predict = model.predict(X_test)

        # Inverse transform to get actual values
        #train_predict = scaler.inverse_transform(train_predict)
        #test_predict = scaler.inverse_transform(test_predict)

        # Reshape input to be [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        #y_train = scaler.inverse_transform([y_train])
        #y_test = scaler.inverse_transform([y_test])


        # Search for the best hyperparameters
        tuner.search(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"The best hyperparameters are: {best_hps.values}")

        model = tuner.hypermodel.build(best_hps)

        # Optionally, retrain on the full training data
        model.fit(X_train, y_train, epochs=30, batch_size=1)

        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform to get actual values
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        y_train = scaler.inverse_transform([y_train])
        y_test = scaler.inverse_transform([y_test])
        
        # Plot the results
        train_plot = np.empty_like(X, dtype=float)
        train_plot[:] = np.nan
        train_plot[time_step:len(train_predict) + time_step] = train_predict[:, 0]

        test_plot = np.empty_like(X, dtype=float)
        test_plot[:] = np.nan
        test_plot[len(train_predict) + (time_step * 2) + 1:len(X) - 1] = test_predict[:, 0]

        plt.figure(figsize=(10, 6))
        plt.plot(X, label='Original Data')
        plt.plot(train_plot, label='Training Prediction')
        plt.plot(test_plot, label='Testing Prediction')
        plt.legend()
        plt.show()
            
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
    elem [steps] = {"results": results}
    elems.append(elem)    
            
    #print('Mean Absolute Error (MAE): ', mae_list, np.mean(mae_list))
    #print('Mean Squared Error (MSE): ', mse_list, np.mean(mse_list))
    #print('R-squared (R²): ', r_squared_list, np.mean(r_squared_list))
    #print('Root Mean Squared Error (RMSE): ', rmse_list, np.mean(rmse_list))
    #print("True Positive: ", true_positive, " False Positive: ", false_positive, " True Negative: ", true_negative, " False Negative: ", false_negative)
    #print("====================================")
    #plt.title("Step: "+str(steps))
    #plt.show()

#filename = sys.argv[1].split(".")[0]
#print(filename)
#with open('../data/'+filename+'.json', 'a') as file:
    #json.dump(elems, file) 