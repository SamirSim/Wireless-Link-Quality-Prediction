import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from pandas import read_csv # type: ignore
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore
from statsmodels.tsa.arima.model import ARIMA # type: ignore
from math import sqrt
from pandas import DataFrame # type: ignore
import time
import warnings
import random
import string
import json
<<<<<<< HEAD
from statsmodels.tsa.stattools import adfuller # type: ignore

random.seed(10)

from itertools import permutations

# Load JSON files (replace with actual file paths)
with open("../data/series-iotj-24h.json", "r") as f:
    data_expe = json.load(f)

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

def check_stationarity(time_series):
    result = adfuller(time_series)
    p_value = result[1]

    if p_value < 0.05:
        print("The time series is likely stationary (p-value =", p_value, ")")
    else:
        print("The time series is likely non-stationary (p-value =", p_value, ")")

for key, values in data_expe.items():
    try:
        print(key)
        check_stationarity(values)
    except:
        pass

# Generate all possible sender-receiver pairs without "m3"
couples = [(int(sender[3:]), int(receiver[3:]))  for sender, receiver in permutations(node_mac_map.keys(), 2)]
=======

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

#couples = [(2, 10), (2, 9), (7, 9), (6, 10), (10, 6), (10, 2), (11, 2), (11,6), (4, 5), (4, 6), (5, 6), (5, 4), (6, 4), (6, 5), (7, 6), (6, 9)]

couples = [(10, 2), (4, 11), (2, 9)]

k = -1
mean_pdr_total = 0

for key, value in data_expe.items():
    mean_pdr = get_mean_pdr(value) / period # This only works because the sending period is the same for all nodes and equal to 1
    #print(key, mean_pdr, len(value))
    mean_pdr_total = mean_pdr_total + mean_pdr
#print("Mean PDR of the network: ", mean_pdr_total/len(data_expe))

>>>>>>> f08aac89880d5104c8998ca951cbcde799f4114e

# Assuming data_expe, data_simu, data_simu_pdr, and couples are defined elsewhere in your code
# couples is a list of tuples where each tuple contains (n, m)

# Adjust the following constants as needed
<<<<<<< HEAD
subplots_per_figure = 25
rows = 5  # Adjust based on the desired subplot arrangement
cols = 5  # Adjust based on the desired subplot arrangement

num_figures = (len(couples) + subplots_per_figure - 1) // subplots_per_figure

#plt.rcParams.update({'font.size': 17})
=======
subplots_per_figure = 3
rows = 1  # Adjust based on the desired subplot arrangement
cols = 3  # Adjust based on the desired subplot arrangement

num_figures = (len(couples) + subplots_per_figure - 1) // subplots_per_figure

link_mse = {'first': ['m3-2_m3-7', 'm3-2_m3-12', 'm3-3_m3-8', 'm3-3_m3-10', 'm3-4_m3-10', 'm3-4_m3-11', 'm3-4_m3-12', 'm3-6_m3-10', 'm3-6_m3-12', 'm3-7_m3-2', 'm3-8_m3-3', 'm3-10_m3-3', 'm3-10_m3-4', 'm3-10_m3-6', 'm3-11_m3-2', 'm3-11_m3-4', 'm3-12_m3-2', 'm3-12_m3-4', 'm3-12_m3-6'], 'second': ['m3-2_m3-3', 'm3-2_m3-4', 'm3-2_m3-5', 'm3-2_m3-6', 'm3-2_m3-8', 'm3-2_m3-9', 'm3-2_m3-10', 'm3-2_m3-11', 'm3-3_m3-2', 'm3-3_m3-4', 'm3-3_m3-5', 'm3-3_m3-6', 'm3-3_m3-7', 'm3-3_m3-9', 'm3-3_m3-11', 'm3-3_m3-12', 'm3-4_m3-2', 'm3-4_m3-3', 'm3-4_m3-5', 'm3-4_m3-6', 'm3-4_m3-7', 'm3-4_m3-8', 'm3-4_m3-9', 'm3-5_m3-2', 'm3-5_m3-3', 'm3-5_m3-4', 'm3-5_m3-6', 'm3-5_m3-7', 'm3-5_m3-8', 'm3-5_m3-9', 'm3-5_m3-10', 'm3-5_m3-11', 'm3-5_m3-12', 'm3-6_m3-2', 'm3-6_m3-3', 'm3-6_m3-4', 'm3-6_m3-5', 'm3-6_m3-7', 'm3-6_m3-8', 'm3-6_m3-9', 'm3-6_m3-11', 'm3-7_m3-3', 'm3-7_m3-4', 'm3-7_m3-5', 'm3-7_m3-6', 'm3-7_m3-8', 'm3-7_m3-9', 'm3-7_m3-10', 'm3-7_m3-11', 'm3-7_m3-12', 'm3-8_m3-2', 'm3-8_m3-4', 'm3-8_m3-5', 'm3-8_m3-6', 'm3-8_m3-7', 'm3-8_m3-9', 'm3-8_m3-10', 'm3-8_m3-11', 'm3-8_m3-12', 'm3-9_m3-2', 'm3-9_m3-3', 'm3-9_m3-4', 'm3-9_m3-5', 'm3-9_m3-6', 'm3-9_m3-7', 'm3-9_m3-8', 'm3-9_m3-10', 'm3-9_m3-11', 'm3-9_m3-12', 'm3-10_m3-2', 'm3-10_m3-5', 'm3-10_m3-7', 'm3-10_m3-8', 'm3-10_m3-9', 'm3-10_m3-11', 'm3-10_m3-12', 'm3-11_m3-3', 'm3-11_m3-5', 'm3-11_m3-6', 'm3-11_m3-7', 'm3-11_m3-8', 'm3-11_m3-9', 'm3-11_m3-10', 'm3-11_m3-12'], 'third': ['m3-12_m3-3', 'm3-12_m3-5', 'm3-12_m3-7', 'm3-12_m3-8', 'm3-12_m3-9', 'm3-12_m3-10', 'm3-12_m3-11'], 'nb_first': 19, 'nb_second': 84, 'nb_third': 7}

plt.rcParams.update({'font.size': 17})
>>>>>>> f08aac89880d5104c8998ca951cbcde799f4114e
for fig_num in range(num_figures):
    fig, axes = plt.subplots(rows, cols, figsize=(27, 8))
    
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing

    for subplot_index in range(subplots_per_figure):
        index = fig_num * subplots_per_figure + subplot_index
        if index >= len(couples):
            break

        n, m = couples[index]
        sender = "m3-" + str(n)
        receiver = "m3-" + str(m)
        key = sender + "_" + receiver
        data_expe_values = data_expe[key]
<<<<<<< HEAD

        y_exp = np.array(data_expe_values)
=======
        data_simu_values = data_simu[key]
        data_simu_pdr_values = data_simu_pdr[key]

        data_simu_values = data_simu_values[:len(data_expe_values)]
        data_simu_pdr_values = data_simu_pdr_values[:len(data_expe_values)]

        x = np.arange(1, len(data_simu_values) + 1)
        y_exp = np.array(data_expe_values)
        y_sim = np.array(data_simu_values)
        y_sim_pdr = np.array(data_simu_pdr_values)
>>>>>>> f08aac89880d5104c8998ca951cbcde799f4114e

        ax = axes[subplot_index]
        
        ax.set_title(key)
        ax.plot(y_exp, label='experiments')
<<<<<<< HEAD
        ax.set_xlabel('Time Window')

        if fig_num == 0 and subplot_index == 0:
            ax.set_ylabel('Number of packets received')
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels)

    plt.show()
    #plt.savefig("Links.pdf", format="pdf", bbox_inches="tight")
    #sys.exit(0)
=======
        ax.plot(y_sim, label='simulation')
        ax.plot(y_sim_pdr, label='simulation_pdr')
        ax.set_xlabel('Time Window', fontsize=19)

        if fig_num == 0 and subplot_index == 0:
            ax.set_ylabel('Number of packets received', fontsize=19)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels)

    #plt.show()
    plt.savefig("Links.pdf", format="pdf", bbox_inches="tight")
    sys.exit(0)
>>>>>>> f08aac89880d5104c8998ca951cbcde799f4114e
