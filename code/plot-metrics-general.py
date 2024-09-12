import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
import json
import time
from sklearn.metrics import roc_curve, auc # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc # type: ignore
import numpy as np # type: ignore
from scipy.special import softmax # type: ignore
import sys

# ARIMA
result = []

with open('../data/results-arima.json', 'r') as file:
    result = json.load(file)

data = []

# Load the series_list from the file
with open('../data/series_list_customized_p.json', 'r') as file:
    series_list = json.load(file)

data_expe = series_list[0]
cpt = 0
for elem in result:
    # Initialize lists to collect all ground truths and predictions
    all_ground_truths = []
    all_predictions = []

    for key, value in elem.items():
        for key_2, value_2 in value.items():
            print(key_2, value_2)
            for elem in value_2:
                for key_3, value_3 in elem.items():
                    #print(key_3, value_3["true_positive"], key)
                    step = key

                    y_pred = value_3["prediction"]
                    y = data_expe[key_3]

                    mae = value_3["mae"]
                    mse = value_3["mse"]
                    print("MAE: ", mae, " MSE: ", mse, " step: ", step)

                    if mse > 300:
                        print("HERE ", key_3, value_3["prediction"], data_expe[key_3], mean_squared_error(data_expe[key_3][:len(value_3["prediction"])], value_3["prediction"]))
                        #mae = None
                        #mse = None
                        #time.sleep(2)

                    data.append({
                        "Step": int(step),
                        "mae": mae,
                        "mse": mse
                    })

df_arima = pd.DataFrame(data)
df_sorted_arima = df_arima.sort_values(by="Step")
df_sorted_arima = df_sorted_arima[df_sorted_arima["Step"] < 15]


# Complete Regression
filename = '../data/results-regression-comprehensive-models.json'
with open(filename, 'r') as file:
    result = json.load(file)

data = []

for elem in result:
    for key, value in elem.items():
        step = int(key)
        mse_list = []
        maes_list = []
        if step >= 1:
            print("step: ", step)
            for key_2, value_2 in value.items():
                try:
                    for key_3, value_3 in value_2.items():
                        data_expe_values = data_expe[key_3]
                            
                        train_size = int(len(data_expe_values) * 0.66)
                        train, test = data_expe_values[0:train_size], data_expe_values[train_size:]

                        for elem in value_3: # Loop over all regressors
                            regressor = elem["regressor"]
                            data.append({"Step": step, "regressor": regressor, "mse": elem["MSE"], "mae": elem["MAE"]})

                except Exception as e:
                    print("Exception: ", e)
                    print(step, key_2, key_3)
                    pass

df_complete_regression = pd.DataFrame(data)
df_sorted_complete_regression = df_complete_regression.sort_values(by="Step")
df_sorted_complete_regression = df_sorted_complete_regression[df_sorted_complete_regression["Step"] < 15]

regressors = ["randomForestRegressor", "svr", "gradientBoostingRegressor", "decisionTreeRegressor", "adaBoostRegressor", "extraTreesRegressor"]

random_forest = df_sorted_complete_regression[df_sorted_complete_regression["regressor"] == "RandomForestRegressor()"]
print("here: ", random_forest)
svr = df_sorted_complete_regression[df_sorted_complete_regression["regressor"] == "SVR()"]
gradient_boosting = df_sorted_complete_regression[df_sorted_complete_regression["regressor"] == "GradientBoostingRegressor()"]
decision_tree = df_sorted_complete_regression[df_sorted_complete_regression["regressor"] == "DecisionTreeRegressor()"]
ada_boost = df_sorted_complete_regression[df_sorted_complete_regression["regressor"] == "AdaBoostRegressor()"]
extra_trees = df_sorted_complete_regression[df_sorted_complete_regression["regressor"] == "ExtraTreesRegressor()"]

# Create violin plots for accuracy and recall
plt.figure(figsize=(7, 6))
plt.rcParams.update({'font.size': 13})

#plt.subplot(2, 1, 1)
# Plot the violin plots for each dataset
# Add a category column to each dataframe
random_forest['Category'] = 'Random Forest'
svr['Category'] = 'SVR'
gradient_boosting['Category'] = 'Gradient Boosting'
decision_tree['Category'] = 'Decision Tree'
ada_boost['Category'] = 'Ada Boost'
extra_trees['Category'] = 'Extra Trees'
df_sorted_arima['Category'] = 'ARIMA'

df_sorted_arima['rmse'] = np.sqrt(df_sorted_arima['mse'])
random_forest['rmse'] = np.sqrt(random_forest['mse'])
svr['rmse'] = np.sqrt(svr['mse'])
gradient_boosting['rmse'] = np.sqrt(gradient_boosting['mse'])
decision_tree['rmse'] = np.sqrt(decision_tree['mse'])
ada_boost['rmse'] = np.sqrt(ada_boost['mse'])
extra_trees['rmse'] = np.sqrt(extra_trees['mse'])

# Concatenate the dataframes
df_combined = pd.concat([random_forest, svr, gradient_boosting, decision_tree, ada_boost, extra_trees, df_sorted_arima])

# Create the violin plot
#sns.violinplot(x="Step", y="mse", hue="Category", data=df_combined, palette={"ARIMA": "red", "Random Forest": "blue", "SVR": "green", "Gradient Boosting": "orange", "Decision Tree": "purple", "Ada Boost": "brown", "Extra Trees": "pink"})
sns.boxplot(x="Step", y="rmse", hue="Category", data=df_combined, palette={"ARIMA": "red", "Random Forest": "blue", "SVR": "green", "Gradient Boosting": "orange", "Decision Tree": "purple", "Ada Boost": "brown", "Extra Trees": "pink"})
plt.yscale('log')

# Add title and labels
plt.title('RMSE')
plt.xlabel('Prediction Step')
plt.ylabel('RMSE')
#plt.legend(loc='upper right')

plt.grid()
"""
plt.subplot(2, 1, 2)
#sns.violinplot(x="Step", y="mae", hue="Category", density_norm="count", data=df_combined, cut=0, palette={"ARIMA": "red", "Random Forest": "blue", "SVR": "green", "Gradient Boosting": "orange", "Decision Tree": "purple", "Ada Boost": "brown", "Extra Trees": "pink"})
sns.boxplot(x="Step", y="mae", hue="Category", data=df_combined, palette={"ARIMA": "red", "Random Forest": "blue", "SVR": "green", "Gradient Boosting": "orange", "Decision Tree": "purple", "Ada Boost": "brown", "Extra Trees": "pink"})
plt.yscale('log')

# Add title and labels
plt.title('MAE')
plt.xlabel('Step')
plt.ylabel('MAE')

plt.tight_layout()

plt.grid()
"""
plt.show()

# Regression
filename = '../data/results-best-regression.json'
with open(filename, 'r') as file:
    result = json.load(file)

data = []

# Load the series_list from the file
with open('../data/series_list_customized_p.json', 'r') as file:
    series_list = json.load(file)

data_expe = series_list[0]
cpt = 0
for elem in result:
    # Initialize lists to collect all ground truths and predictions
    all_ground_truths = []
    all_predictions = []

    for key, value in elem.items():
        if int(key) > 10:
            break
        for key_2, value_2 in value.items():
            try:
                for elem in value_2:
                    for key_3, value_3 in elem.items():
                        step = key
                        data_ = data_expe[key_3]
                        size = int(len(data) * 0.66) # First 2/3 of the data are used for training, and the rest is used for testing
                        test = data_[size:len(data_)]

                        y_pred = value_3["predictions"]
                        test = test[:len(y_pred)]

                        mae = value_3["MAE"]
                        mse = value_3["MSE"]
                        print("MAE: ", mae, " MSE: ", mse, " step: ", step)
                        if mse > 300:
                            print("HERE ", key_3, value_3["predictions"], data_expe[key_3], mean_squared_error(data_expe[key_3][:len(value_3["predictions"])], value_3["predictions"]))

                        data.append({
                            "Step": int(step),
                            "mae": mae,
                            "mse": mse
                        })

            except Exception as e:
                print("Exception: ", e)
                pass

df_regression = pd.DataFrame(data)
df_sorted_regression = df_regression.sort_values(by="Step")

# Adaptive Regression
filename = '../data/results-regression-comprehensive-models.json'
with open(filename, 'r') as file:
    result = json.load(file)

data = []

# Load the series_list from the file
with open('../data/series_list_customized_p.json', 'r') as file:
    series_list = json.load(file)

data_expe = series_list[0]
regressors_list = []

cpt = 0
for elem in result:
    # Initialize lists to collect all ground truths and predictions
    all_ground_truths = []
    all_predictions = []

    for key, value in elem.items():
        step = int(key)
        mse_list = []
        maes_list = []
        if step <= 10:
            print("step: ", step)
            best_predictions = []
            for key_2, value_2 in value.items():
                try:
                    for key_3, value_3 in value_2.items():
                        final_predictions = []
                        data_expe_values = data_expe[key_3]
                            
                        train_size = int(len(data_expe_values) * 0.66)
                        train, test = data_expe_values[0:train_size], data_expe_values[train_size:]

                        print("train: ", train, train[-1])
                        print("test: ", test)
                        #time.sleep(2)
                        # Add the last element of the training set to the test set
                        test = np.insert(test, 0, train[-1])

                        print("test: ", test)
                        #time.sleep(1)
                        index = 0
                        while index <= len(test):
                            min_mse = float('inf')
                            best_prediction = []
                            for elem in value_3: # Loop over all regressors
                                elem_ = []
                                regressor = elem["regressor"]
                                try:
                                    for i in range(0, step):
                                        elem_.append(elem["predictions"][index+i]) # Changed here from i+1 to i
                                        print(regressor, index, test[index+i], elem["predictions"][index+i]) # Changed here from i+1 to i
                                        #time.sleep(1)
                                    mse = mean_squared_error(test[index:index+step], elem_)
                                    
                                    if mse < min_mse:
                                        min_mse = mse
                                        best_prediction = [v for v in elem_]
                                        best_regressor = regressor
                                        print("Current best regressor: ", best_regressor, " best prediction: ", best_prediction)
                                    print("MSE: ", mse)
                                except Exception as e:
                                    print("Exception1: ", e)
                                    pass
                            
                            for v in best_prediction:
                                final_predictions.append(v)
                            print("here ", final_predictions)
                            #time.sleep(1)
                            regressors_list.append(best_regressor)

                            index += step
                        print("Final predictions: ", final_predictions)
                        
                        if step == 1:
                            print("here")
                            print(len(final_predictions), len(test))
                            print(final_predictions)
                            print(regressors_list)
                            #time.sleep(5)
                        mse = mean_squared_error(test[:len(final_predictions)], final_predictions)
                        mae = mean_absolute_error(test[:len(final_predictions)], final_predictions)

                        if mse > 100:
                            print("Here MSE too big: MSE: ", mse, " Key: ", key_3, " Step: ", step)
                            print("Test: ", test[:len(final_predictions)])
                            print("Predictions: ", final_predictions)
                            #time.sleep(5)
                        mse_list.append(mse)
                        maes_list.append(mae)
                        print("Final MSE: ", mse, "Final MAE: ", mae)

                except Exception as e:
                    print("Exception: ", e)
                    print(step, key_2, key_3)
                    #time.sleep(10)
                    #print(all_predictions)
                    pass
                        # Convert lists to numpy arrays
    print("Step: ", step)
    if step == 3:
        print("here")
        print(mse_list)
        #time.sleep(5)

    for i in range(len(mse_list)):
        data.append({"Step": step, "mse": mse_list[i], "mae": maes_list[i]})
    #time.sleep(2)

df_adaptive_regression = pd.DataFrame(data)
df_sorted_adaptive_regression = df_adaptive_regression.sort_values(by="Step")

# RNN
filename = '../data/rnn.json'
with open(filename, 'r') as file:
    result = json.load(file)

data = []

data_expe = series_list[0]
cpt = 0
for elem in result:
    # Initialize lists to collect all ground truths and predictions
    all_ground_truths = []
    all_predictions = []

    for key, value in elem.items():
        print(key)
        if key == "Time taken":
            break
        if int(key) > 10:
            break
        #time.sleep(2)
        for key_2, value_2 in value.items():
            try:
                print("here ", key_2)
                #time.sleep(5)
                for key_3, value_3 in value_2.items():
                    for elem_ in value_3:
                        print("monsieur, ", key_3, elem_)
                        print(key_3, key)
                        step = key
                        data_ = data_expe[key_3]
                        size = int(len(data) * 0.66) # First 2/3 of the data are used for training, and the rest is used for testing
                        test = data_[size:len(data_)]

                        y_pred = elem_["predictions"]
                        test = test[:len(y_pred)]

                        #print(key_3, y_pred, test, len(test))

                        mae = elem_["MAE"]
                        mse = elem_["MSE"]
                        print("MAE: ", mae, " MSE: ", mse, " step: ", step)
 
                        data.append({
                            "Step": int(step),
                            "mae": mae,
                            "mse": mse
                        })

            except Exception as e:
                print("Exception: ", e)
                #print(all_predictions)
                pass
                    # Convert lists to numpy arrays
    
df_rnn= pd.DataFrame(data)
df_sorted_rnn = df_rnn.sort_values(by="Step")

# Plotting
plt.figure(figsize=(7, 6))
plt.rcParams.update({'font.size': 13})


#plt.subplot(2, 1, 1)
# Plot the violin plots for each dataset
# Add a category column to each dataframe
df_sorted_adaptive_regression['Category'] = 'Adaptive Regression'
df_sorted_regression['Category'] = 'Regression'
df_sorted_arima['Category'] = 'ARIMA'
df_sorted_rnn['Category'] = 'RNN'

df_sorted_adaptive_regression['rmse'] = np.sqrt(df_sorted_adaptive_regression['mse'])
df_sorted_regression['rmse'] = np.sqrt(df_sorted_regression['mse'])
df_sorted_arima['rmse'] = np.sqrt(df_sorted_arima['mse'])
df_sorted_rnn['rmse'] = np.sqrt(df_sorted_rnn['mse'])

# Concatenate the dataframes
df_combined = pd.concat([df_sorted_adaptive_regression, df_sorted_regression, df_sorted_rnn])

# Create the violin plot
"""
#sns.violinplot(x="Step", y="mae", hue="Category", data=df_combined, cut=0, palette={"Adaptive Regression": "blue", "Regression": "green"})
sns.boxplot(x="Step", y="mae", hue="Category", data=df_combined, palette={"Adaptive Regression": "blue", "Regression": "green", "RNN": "red"})
plt.yscale('log')

# Add title and labels
plt.title('MAE')
plt.xlabel('Step')
plt.ylabel('MAE')
plt.grid()
"""
#plt.subplot(2, 1, 2)

#sns.violinplot(x="Step", y="mse", hue="Category", data=df_combined, cut=0, palette={"Adaptive Regression": "blue", "Regression": "green"})
sns.boxplot(x="Step", y="rmse", hue="Category", data=df_combined, palette={"Adaptive Regression": "blue", "Regression": "green", "RNN": "red"})
plt.yscale('log')

# Add title and labels
plt.title('RMSE')
plt.xlabel('Prediction Step')
plt.ylabel('RMSE')
plt.grid()


plt.legend()
plt.tight_layout()


plt.savefig("Adaptive-Models-RMSE.pdf", format="pdf", bbox_inches="tight")
sys.exit(0)
plt.show()
