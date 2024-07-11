import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
import json
import time
from sklearn.metrics import roc_curve, auc # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc # type: ignore
import pmdarima as pm # type: ignore
from statsmodels.tsa.stattools import adfuller # type: ignore
import numpy as np # type: ignore
from scipy.special import softmax # type: ignore
import sys

def plot_series(series, title="Time Series"):
    plt.figure(figsize=(10, 6))
    plt.plot(series)
    plt.title(title)
    plt.show()

def is_stationary(time_series, significance_level=0.05):
    result = adfuller(time_series)
    p_value = result[1]
    return p_value < significance_level

def auto_differencing(time_series):
    # Use pmdarima's ndiffs to find the optimal number of differences
    d = pm.arima.ndiffs(time_series, test='adf')
    diff_series = time_series
    for _ in range(d):
        diff_series = np.diff(diff_series, n=1)

    return diff_series, d

# ARIMA
result = []

with open('../data/results-arima.json', 'r') as file:
    result = json.load(file)

data = []

# Load the series_list from the file
with open('../data/series_list_customized_p.json', 'r') as file:
    series_list = json.load(file)

link_mse = {"first": [], "second": [], "third": [], "nb_first": 0, "nb_second": 0, "nb_third": 0}

data_expe = series_list[0]

"""
cpt = 0
for elem in result:
    # Initialize lists to collect all ground truths and predictions
    all_ground_truths = []
    all_predictions = []

    for key, value in elem.items():
        if int(key) > 1:
            continue
        
        for key_2, value_2 in value.items():
            #print(key_2, value_2)
            for elem in value_2:
                for key_3, value_3 in elem.items():
                    #print(key_3, value_3["true_positive"], key)
                    step = key

                    y_pred = value_3["prediction"]
                    y = data_expe[key_3]

                    mae = value_3["mae"]
                    mse = value_3["mse"]
                    #print("MAE: ", mae, " MSE: ", mse, " step: ", step)

                    test_data = y[-len(y_pred):]
                    # ADF test
                    result = adfuller(y)
                    #print('ADF Statistic: %f' % result[0])
                    #print('p-value: %f' % result[1])
                    adf_statistic, p_value, _, _, critical_values, _ = result

    
                    # Check if p-value is less than the significance level
                    significance_level=0.05
                    is_stationary_ = p_value < significance_level
                    #print('Variance: ', np.var(y))

                    

                    if mae < 2:
                        #print("Link: ", key_3)
                        link_mse["first"].append(key_3)
                    elif mae < 5:
                        link_mse["second"].append(key_3)
                    else:
                        link_mse["third"].append(key_3)

                    if mse > 3000000:
                        print("Link: ", key_3)
                        print("The time series is ", "stationary" if is_stationary_ else "not stationary")
                        pred_plot = np.empty_like(y, dtype=float)
                        pred_plot[:] = np.nan
                        for i in range(len(y_pred)):
                            pred_plot[-len(y_pred)+i] = y_pred[i]
                        print('MSE: ', mse, ' MAE: ', mae)
                        # Compute variance of the series
                        print('Variance: ', np.var(y))

                        # Plot the series
                        plt.figure(figsize=(12, 6))
                        plt.plot(y, label='True')
                        plt.plot(pred_plot, label='Predicted')
                        plt.legend()
                        plt.title(key_3)
                        plt.show()

                        time_series_data = np.array(y).cumsum()

                        # Perform auto differencing
                        diff_series, num_diffs = auto_differencing(time_series_data)
                        
                        # Plot the differenced series
                        plot_series(diff_series, title=f"Differenced Series (d={num_diffs})")
                        
                        # Check if the differenced series is stationary
                        if is_stationary(diff_series):
                            print(f"The differenced time series is stationary after {num_diffs} differences.")
                        else:
                            print(f"The differenced time series is not stationary after {num_diffs} differences.")

                    if mse > 300:
                        pass
                        #print("HERE ", key_3, value_3["prediction"], data_expe[key_3], mean_squared_error(data_expe[key_3][:len(value_3["prediction"])], value_3["prediction"]))
                        #mae = None
                        #mse = None
                        #time.sleep(2)

                    data.append({
                        "Step": int(step),
                        "mae": mae,
                        "mse": mse
                    })

link_mse["nb_first"] = len(link_mse["first"])
link_mse["nb_second"] = len(link_mse["second"])
link_mse["nb_third"] = len(link_mse["third"])

print("Link MSE: ", link_mse)

time.sleep(5)

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
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
# Plot the violin plots for each dataset
# Add a category column to each dataframe
random_forest['Category'] = 'Random Forest'
svr['Category'] = 'SVR'
gradient_boosting['Category'] = 'Gradient Boosting'
decision_tree['Category'] = 'Decision Tree'
ada_boost['Category'] = 'Ada Boost'
extra_trees['Category'] = 'Extra Trees'
df_sorted_arima['Category'] = 'ARIMA'

# Concatenate the dataframes
df_combined = pd.concat([random_forest, svr, gradient_boosting, decision_tree, ada_boost, extra_trees, df_sorted_arima])

# Create the violin plot
sns.violinplot(x="Step", y="mse", hue="Category", density_norm="count", data=df_combined, cut=0, palette={"ARIMA": "red", "Random Forest": "blue", "SVR": "green", "Gradient Boosting": "orange", "Decision Tree": "purple", "Ada Boost": "brown", "Extra Trees": "pink"})
#sns.boxplot(x="Step", y="mse", hue="Category", data=df_combined, palette={"ARIMA": "red", "Random Forest": "blue", "SVR": "green", "Gradient Boosting": "orange", "Decision Tree": "purple", "Ada Boost": "brown", "Extra Trees": "pink"})

# Add title and labels
plt.title('MSE')
plt.xlabel('Step')
plt.ylabel('MSE')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
sns.violinplot(x="Step", y="mae", hue="Category", density_norm="count", data=df_combined, cut=0, palette={"ARIMA": "red", "Random Forest": "blue", "SVR": "green", "Gradient Boosting": "orange", "Decision Tree": "purple", "Ada Boost": "brown", "Extra Trees": "pink"})
#sns.boxplot(x="Step", y="mae", hue="Category", data=df_combined, palette={"ARIMA": "red", "Random Forest": "blue", "SVR": "green", "Gradient Boosting": "orange", "Decision Tree": "purple", "Ada Boost": "brown", "Extra Trees": "pink"})

# Add title and labels
plt.title('MAE')
plt.xlabel('Step')
plt.ylabel('MAE')

plt.tight_layout()

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
"""

# Adaptive Regression
filename = '../data/results-regression-comprehensive-models.json'
with open(filename, 'r') as file:
    result = json.load(file)

data = []

cluster_data = {"1": [], "2": [], "3": [], "5": [], "7": [], "10": [], "15": [], "20": []}

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
        if step >= 1:
            print("step: ", step)
            best_predictions = []
            for key_2, value_2 in value.items():
                try:
                    for key_3, value_3 in value_2.items():
                        res = {}
                        final_predictions = []
                        data_expe_values = data_expe[key_3]
                            
                        train_size = int(len(data_expe_values) * 0.66)
                        train, test = data_expe_values[0:train_size], data_expe_values[train_size:]

                        index = 0
                        while index <= len(test):
                            min_mse = float('inf')
                            best_prediction = []
                            for elem in value_3: # Loop over all regressors
                                elem_ = []
                                regressor = elem["regressor"]
                                try:
                                    for i in range(0, step):
                                        elem_.append(elem["predictions"][index+i])
                                        print(regressor, index, test[index+i], elem["predictions"][index+i+1])
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
                            print(test[:len(final_predictions)])
                            print(final_predictions)
                            #time.sleep(5)
                        mse = mean_squared_error(test[:len(final_predictions)], final_predictions)
                        mae = mean_absolute_error(test[:len(final_predictions)], final_predictions)

                        
                        res[key_3] = {"mse": mse, "mae": mae}

                        if mae < 2:
                        #print("Link: ", key_3)
                            link_mse["first"].append(key_3)
                        elif mae < 5:
                            link_mse["second"].append(key_3)
                        else:
                            link_mse["third"].append(key_3)

                        if mse > 100:
                            print("Here MSE too big: MSE: ", mse, " Key: ", key_3, " Step: ", step)
                            print("Test: ", test[:len(final_predictions)])
                            print("Predictions: ", final_predictions)
                            #time.sleep(5)
                        mse_list.append(mse)
                        maes_list.append(mae)
                        print("Final MSE: ", mse, "Final MAE: ", mae)

                        cluster_data[str(step)].append(res)  
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

link_mse["nb_first"] = len(link_mse["first"])
link_mse["nb_second"] = len(link_mse["second"])
link_mse["nb_third"] = len(link_mse["third"])

print("Link MSE: ", link_mse)

filename = "cluster_data.json"
print(cluster_data)
#print(filename)
with open('../data/'+filename+'.json', 'a') as file:
    json.dump(cluster_data, file)

exit()
df_adaptive_regression = pd.DataFrame(data)
df_sorted_adaptive_regression = df_adaptive_regression.sort_values(by="Step")

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
# Plot the violin plots for each dataset
# Add a category column to each dataframe
df_sorted_adaptive_regression['Category'] = 'Adaptive Regression'
df_sorted_regression['Category'] = 'Regression'
df_sorted_arima['Category'] = 'ARIMA'

# Concatenate the dataframes
df_combined = pd.concat([df_sorted_adaptive_regression, df_sorted_regression])

# Create the violin plot
#sns.violinplot(x="Step", y="mse", hue="Category", data=df_combined, cut=0, palette={"Adaptive Regression": "blue", "Regression": "green"})
sns.boxplot(x="Step", y="mse", hue="Category", data=df_combined, palette={"Adaptive Regression": "blue", "Regression": "green"})
plt.yscale('log')

# Add title and labels
plt.title('MSE')
plt.xlabel('Step')
plt.ylabel('MSE')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
#sns.violinplot(x="Step", y="mae", hue="Category", data=df_combined, cut=0, palette={"Adaptive Regression": "blue", "Regression": "green"})
sns.boxplot(x="Step", y="mae", hue="Category", data=df_combined, palette={"Adaptive Regression": "blue", "Regression": "green"})
plt.yscale('log')

# Add title and labels
plt.title('MAE')
plt.xlabel('Step')
plt.ylabel('MAE')

plt.tight_layout()

plt.show()
