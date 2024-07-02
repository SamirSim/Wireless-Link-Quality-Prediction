from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor # type: ignore
from sklearn.tree import DecisionTreeRegressor # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.svm import SVR # type: ignore
from sklearn.neighbors import KNeighborsRegressor # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc # type: ignore
from scipy.special import softmax # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import time

np.random.seed(10)

def create_sequences(data, window_step):
    X, y = [], []
    for i in range(len(data) - window_step):
        X.append(data[i:(i + window_step)])
        y.append(data[i + window_step])
    return np.array(X), np.array(y)

# Generate random numpy array
#data = range(1, 16, 1)
data = [np.random.randint(10) for i in range(15)]

X = [2, 1, 0, 3, 2, 4, 1, 4, 4, 3, 7, 8, 7, 3, 2, 0, 1, 0, 3, 0, 2, 0, 4, 1, 1, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 28, 32, 32, 37, 27, 28, 29, 32, 25, 25, 31, 36, 38, 32, 34, 29, 32, 35, 33, 35, 35, 33, 38, 34, 35, 32, 42, 43, 41, 31, 37, 40, 38, 44, 37, 31, 41, 41, 43, 40, 44, 38, 34, 40, 39, 32, 33, 39, 39, 38, 41, 43, 40, 36, 43, 41, 39, 38, 41, 41, 41, 37, 34, 44, 37, 38, 40, 36, 38, 40, 36, 40, 42, 43, 39, 41, 34, 44, 42, 47, 38, 40, 33, 36, 41, 38, 33, 36, 36, 35, 33, 32, 42, 33, 35, 35, 38, 39, 26, 31, 42, 37, 35, 37, 37, 37, 44, 37, 34, 44, 39, 39, 44, 43, 39, 43, 39, 40, 40, 40, 41, 38, 38, 35, 40, 31, 44, 37, 36, 44, 39, 40, 18, 24, 20, 15, 15, 5, 37, 39, 40, 37, 34, 37, 34, 34, 36, 34, 35, 44, 37, 31, 37, 37, 39, 39, 44, 40, 41, 41, 37, 39, 41, 39, 39, 35, 39, 36, 37, 42, 40, 42, 40, 40, 41, 40, 35, 41, 40, 41, 44, 42, 36, 42, 39, 42, 44, 37, 41, 42, 47, 44, 40, 40, 46, 43, 37, 43, 42, 45, 38, 37, 39, 39, 42, 42, 44, 42]
#X = data
size = int(len(X) * 0.66) # First 2/3 of the data are used for training, and the rest is used for testing
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = []

print("data: ", X)
print("train: ", train)
print("test: ", test)

randomForestRegressor = RandomForestRegressor(n_estimators = 100, random_state = 1)

#regressors = [randomForestRegressor, linearRegressor, svr, gradientBoostingRegressor, decisionTreeRegressor, adaBoostRegressor, extraTreesRegressor, kNeighborsRegressor]
regressors = [randomForestRegressor]

prediction_step = 3

for regressor in regressors:
    window_steps = [7]
    for window_step in window_steps:
        # Prepare sequences
        X_train, y_train = create_sequences(train, window_step)
        #print("train sequence: ", X_train, y_train)
        X_test, y_test = create_sequences(test, window_step)
        #print("test sequence: ", X_test, y_test)
        #time.sleep(1)
        regressor.fit(X_train, y_train)

        #print("element to add: ", train[-window_step:])
        test = train[-window_step:] + test
        print("new test: ", test)
        #print("new test sequence: ", X_test, y_test)
        #time.sleep(2)
        # List to store predictions
        new_test = [v for v in test]
        #window_step = window_step
        # Iterate over the test set
        inputs = []
        index = 0
        predictions = []

        while index < len(test):
            new_test = [v for v in test]
            # Refit the model every `step` predictions
            try:
                for i in range(index, index+prediction_step):
                    input_window = new_test[i:window_step+i]
                    #print("input window: ", input_window)
                    #time.sleep(1)
                    reshaped_input_window = np.array(input_window).reshape(1, -1)
                    print("reshaped input window: ", reshaped_input_window)
                    inputs.append(reshaped_input_window)
                    # Predict the next value
                    predicted_value = regressor.predict(reshaped_input_window)
                    print("predicted value: ", predicted_value)
                    predictions.append(predicted_value)
                        
                    #print("new test: ", new_test)
                    # Replace the element in the TestSet with the predicted value
                    new_test[window_step+i] = predicted_value[0]
                    i = i + 1
                    #time.sleep(2)

                #print("test: ", test, index, window_step)
                refit_data = test[index:index+prediction_step+window_step]
                print("refit data: ", refit_data)
                #print("refit data start: ", refit_data_start)
                #print("refit data end: ", refit_data_end)
                #print("inputs: ", inputs)

                #new_train = 
                            
                # Prepare data for refitting
                X_train, y_train = create_sequences(refit_data, window_step)
                print("Refit with: X train: ", X_train, "y train: ", y_train)
                #time.sleep(2)
                        
                # Refit the model
                regressor.fit(X_train, y_train)
            except Exception as e:
                print("Refit failed with error: ", e)
                break # list index out of range
            index = index + prediction_step
    
print("predictions: ", predictions)