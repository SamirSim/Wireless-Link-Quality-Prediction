import seaborn as sns

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import random
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import pylab 
import scipy.stats as stats
from math import sqrt
from pandas import DataFrame
import time
import warnings
from fitter import Fitter, get_common_distributions, get_distributions
from scipy.stats import norm, shapiro, ks_2samp
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

log_filename = "../data/merged-data.rawdata"

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
    
    timestamp, node_id, event_type, message_id = line.split(";")

    if node_id not in list_nodes:
        list_nodes.append(node_id)

    if event_type == "Sending broadcast":
        sender = node_id
        message_id = message_id.strip() 
        if message_id not in communication_dict:
            communication_dict[message_id] = {"sender": sender, "receivers_list": []}

    elif event_type == "Received":
        receiver = node_id
        message_id = message_id.strip()  # Remove leading space

        #print(communication_dict)
        if message_id in communication_dict:
            communication_dict[message_id]["receivers_list"].append(receiver)
        else:
            print("Message ", message_id," received before sending")

# Print the resulting dictionary
#print(communication_dict)

temp_dict = {}

for key, value in communication_dict.items():
    sender = value.get("sender")
    if sender not in temp_dict:
        temp_dict[sender] = []
    temp_dict[sender].append(value.get("receivers_list"))

#print(series_dict)
first_node = 95
nb_nodes = 5

series_dict = {}

#print(list_nodes)

for key, value in temp_dict.items():
    #print(key,value)
    sender = key
    cpt = 1

    node_ids = []
    # Loop over all the other nodes
    node = first_node
    while cpt <= nb_nodes:
        if "m3-"+str(node) in list_nodes:
            node_ids.append(node)
            receiver = "m3-"+str(node)
            new_key = sender+'_'+receiver
            series_dict[new_key] = []
            for list in value:
                if receiver in list:
                    series_dict[new_key].append(1)
                else:
                    series_dict[new_key].append(0)
            cpt = cpt + 1
            node = node + 1
        else:
            node = node + 1

#print(series_dict)

random_senders = random.sample(node_ids, 3)
random_receivers = random.sample(node_ids, 3)

Position = range(1,10)

k = -1

pdr_data_list = {}

for n in random_senders:
    for m in random_receivers:
        k = k + 1
        #print(k)
        sender = "m3-"+str(n)
        receiver = "m3-"+str(m)

        for elem in series_dict:
            break
            #print(elem)
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
                #print("Exception occured")
                break

        #print(pdr_data, len(pdr_data))

        #time.sleep(3)
        pdr_data_list[key] = pdr_data

        data = pdr_data

        fig = plt.figure(1)
        fig.tight_layout(pad=0.5)

        #print (k, Position[k])
        # Fit a normal distribution to the data:
        mu, std = norm.fit(data)

        print("Fitted mean and standard deviation: ", mu,std)
        p = norm.pdf(data, mu, std)
        ax = fig.add_subplot(3,3,Position[k])

        # Shapiro Test
        print("Shapiro Test: ", shapiro(data))

        # Create QQ Plot
        #stats.probplot(data, dist="norm", plot=pylab)
        
        # Fit kernel density estimation to the original data
        bandwidth = np.arange(0.05, 2, .05)
        kde = KernelDensity(kernel='gaussian')
        grid = GridSearchCV(kde, {'bandwidth': bandwidth})
        grid.fit(np.array(data).reshape(-1,1))

        kde = grid.best_estimator_
        log_dens = kde.score_samples(np.array(data).reshape(-1,1))

        # Generate synthetic data using kernel density estimation
        synthetic_data = kde.sample(len(data)).flatten()

        # print(synthetic_data)

        #time.sleep(15)

        #sns.displot(data=data, ax=ax, kind="hist")
        try:
            sns.distplot(data, fit_kws={'color':'red', 'label':'fit'}, fit=norm, kde=True)
            sns.distplot(synthetic_data, label="KDE")
            plt.legend()
        except:
            print("Exception occured")
            continue
        #sns.kdeplot(data, ax=ax)
        #x = np.linspace(0, len(p))
        #ax.plot(x, p, color='red')

        #time.sleep(5)

        # Fit Gaussian distribution and plot
        #sns.distplot(data, fit=norm, kde=False)
        ax.set_title(key)
        #ax.plot(y)      # Or whatever you want in the subplot

        # Add labels to the plot
        plt.xlabel('Data Values')
        plt.ylabel('Density')

        # Initialize the fitter object
        #f = Fitter(data)

        # Fit common distributions
        #f.fit(get_common_distributions())

        # Print the summary of the fitted distributions
        #print(f.summary())

        #plt.show()
        """
        x = np.arange(1, len(data) + 1) 
        y = np.array(data)
        
        fig = plt.figure(1)
        fig.tight_layout(pad=0.5)

        print (k, Position[k])
        ax = fig.add_subplot(3,3,Position[k])
        ax.set_title(key)
        ax.plot(y)      # Or whatever you want in the subplot

        #plt.show()
        # plotting
        #plt.title(key) 
        #plt.xlabel("X axis") 
        #plt.ylabel("Y axis") 
        #plt.plot(x, y, color ="blue") 
        #plt.show()
plt.show()
"""
history = []
# Smirnov Kolmogorov
for key1, value1 in pdr_data_list.items():
    for key2, value2 in pdr_data_list.items():
        if key1 != key2 and str(key2)+str(key1) not in history:
            statistic, p_value = ks_2samp(value1, value2)
            print(key1, " and ", key2, " : ", p_value) # Add the other link
            history.append(str(key1)+str(key2))

pylab.show()
plt.show()
plt.legend()