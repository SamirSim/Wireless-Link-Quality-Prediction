import sys
# LSTM for international airline passengers problem with window regression framing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

log_filename = "telemetry.rawdata"

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
first_node = 100
nb_nodes = 20

series_dict = {}

#print(list_nodes)

for key, value in temp_dict.items():
    #print(key,value)
    sender = key
    cpt = 1

    # Loop over all the other nodes
    node = first_node
    while cpt < nb_nodes:
        if "m3-"+str(node) in list_nodes:
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
            
sender = "m3-110"
receiver = "m3-111"

key = sender+"_"+receiver
data = series_dict[key]

pdr_data = []

i = 0
window_size = 5

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

# Testing ARIMA
x = np.arange(1, 50) 
y = np.array(pdr_data[:49])
 
# plotting
plt.title("Line graph") 
plt.xlabel("X axis") 
plt.ylabel("Y axis") 
plt.scatter(x, y, color ="blue") 
plt.show()

#data = x
# fit model
model = ARIMA(data, order=(10,1,1))
model_fit = model.fit()

# summary of fit model
print(model_fit.summary())
# line plot of residuals
#residuals = DataFrame(model_fit.resid)
#residuals.plot()
#plt.show()
# density plot of residuals
#residuals.plot(kind='kde')
#plt.show()
# summary stats of residuals
#print(residuals.describe())

# split into train and test sets
X = data
size = int(len(X) * 0.66) # First 2/3 of the data are used for training, and the rest is used for testing
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = []
# walk-forward validation
print(test, size)

for t in range(len(test)):
    model = ARIMA(history, order=(5,1,1))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

