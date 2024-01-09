import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.activations import sigmoid
import matplotlib.pyplot as plt

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

# Example Binary Data
data_length = np.array(data).size
#binary_data = np.random.choice([0, 1], size=data_length)

# Generate synthetic timestamps
timestamps = pd.date_range(start='2023-01-01', periods=data_length, freq='D')

# Create DataFrame
df = pd.DataFrame({'timestamp': timestamps, 'label': data})
df['timestamp_seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

# Prepare sequences for LSTM
sequence_length = 5  # Adjust based on your requirements
X, y = [], []

for i in range(len(df) - sequence_length):
    X.append(df.index[i:i+sequence_length].values.astype(np.int64))
    y.append(df.iloc[i+sequence_length]['label'])

X = np.array(X).reshape(-1, sequence_length, 1)  # Reshape for LSTM input
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X, y)

# Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1, activation=sigmoid))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=1, validation_split=0.2)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Predict on the test set and apply threshold
#y_pred_probs = model.predict(X_test)
#y_pred = (y_pred_probs > 0.5).astype(int)

# Predict probabilities
probabilities = model.predict(X_test)

# Apply threshold (e.g., 0.5) to get binary predictions
predictions = (probabilities > 0.75).astype(int)

print(predictions)

y_pred = model.predict(X_test)

y_pred_labels = (y_pred > 0.7).astype(int)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Real Data', marker='o')
plt.plot(y_pred_labels, label='Predicted Data', marker='x')
plt.title('Real vs. Predicted Test Data')
plt.xlabel('Sample Index')
plt.ylabel('Binary Label')
plt.legend()
plt.show()