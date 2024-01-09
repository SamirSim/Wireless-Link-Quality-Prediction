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
            
sender = "m3-119"
receiver = "m3-102"

key = sender+"_"+receiver
data = series_dict[key] 

# fix random seed for reproducibility
tf.random.set_seed(7)
# load the dataset
dataframe = read_csv('passengers.csv', usecols=[1], engine='python')
#dataset = dataframe.values
dataset = np.array([[x] for x in data])
#dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

print(np.array(data), dataset)
print(np.array(data).shape, dataset.shape)

#print(data.shape, dataset.shape)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print(trainX)

#while 1:
#    continue


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

print(trainPredict, testPredict)

# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

M_TEST = testX.shape[0] 
predict_x=model.predict(testX) 
classes_x=np.argmax(predict_x,axis=1)
#y_hat = model.predict_classes(testX, batch_size=M_TEST, verbose=1)
#score = sum(y_hat == y_test) / len(y_test)
#print(f'Prediction accuracy = {score*100}%')
#index = pd.date_range(start='2017-01-02', end='2018-06-19', freq='B')
#for i in range(predict_x.shape[0]):
print("predict: ", predict_x)
print("y: ", trainX)



# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()