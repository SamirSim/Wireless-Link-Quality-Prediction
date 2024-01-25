import sys
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.arima.model import ARIMA
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

# Example data
data = [
    "1702983920.039034;m3-76;Sending broadcast;a1df916a",
    "1702983920.043096;m3-79;Received;a1df916a",
    "1702983920.042578;m3-71;Received;a1df916a",
    "1702983920.045477;m3-74;Received;a1df916a",
    "1702983920.048265;m3-75;Received;a1df916a",
    "1702983920.052321;m3-73;Received;a1df916a",
    "1702983920.055296;m3-72;Received;a1df916a",
    "1702983920.055685;m3-77;Received;a1df916a",
    "1702983920.055728;m3-78;Received;a1df916a",
    "1702983928.581563;m3-122;Sending broadcast;bbe543d3",
    "1702983928.585259;m3-126;Received;bbe543d3",
    "1702983928.585897;m3-115;Received;bbe543d3",
    "1702983928.586890;m3-120;Received;bbe543d3",
    "1702983928.589152;m3-114;Received;bbe543d3",
    "1702983920.039034;m3-76;Sending broadcast;a1df917a",
    "1702983920.043096;m3-89;Received;a1df917a",
    "1702983920.042578;m3-81;Received;a1df917a",
    "1702983920.045477;m3-84;Received;a1df917a",
]

##################### Dataset Building #####################
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
# Data is ready. Select in (sender, receiver) the targeted wireless link            
sender = "m3-119"
receiver = "m3-102"

key = sender+"_"+receiver
data = series_dict[key] 
print(data)

x = np.arange(1, 50) 
y = np.array(series_dict[key][:49])
 
# plotting
plt.title("Line graph") 
plt.xlabel("X axis") 
plt.ylabel("Y axis") 
plt.scatter(x, y, color ="blue") 
plt.show()

#data = x
# fit model
model = ARIMA(data, order=(1,1,0))
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
    model = ARIMA(history, order=(10,0,0))
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

"""
# Applying logistic regression for prediction
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)

# Create classifier using logistic regression
classifier = LogisticRegression()

# Training our model
classifier.fit(X_train, y_train)

# Predicting values using our trained model
y_pred = classifier.predict(X_test)

plt.scatter(X_test, y_test, color = 'red')
plt.plot(sorted(X_test), sorted(y_pred), '--',color = 'blue')
plt.title('Temperature vs purchase decision (Test set)')
plt.xlabel('temperature')
plt.ylabel('purchase or not')
plt.grid()
plt.show()
"""
"""
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
 # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
        # calculate out of sample error
        error = mean_squared_error(test, predictions)
    return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
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

p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(data, p_values, d_values, q_values)
"""