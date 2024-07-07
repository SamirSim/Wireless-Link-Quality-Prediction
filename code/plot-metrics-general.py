import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
import json
import time
from sklearn.metrics import roc_curve, auc # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc # type: ignore
import numpy as np # type: ignore
from scipy.special import softmax # type: ignore

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

# Regression
filename = '../data/results-complete-regression.json'
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
        if step >= 1:
            print("step: ", step)
            best_predictions = []
            for key_2, value_2 in value.items():
                try:
                    for key_3, value_3 in value_2.items():
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
                                        elem_.append(elem["predictions"][index+i+1])
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


# Create violin plots for accuracy and recall
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
# Plot the violin plots for each dataset
# Add a category column to each dataframe
df_sorted_adaptive_regression['Category'] = 'Adaptive Regression'
df_sorted_regression['Category'] = 'Regression'
df_sorted_arima['Category'] = 'ARIMA'

# Concatenate the dataframes
df_combined = pd.concat([df_sorted_adaptive_regression, df_sorted_regression, df_sorted_arima])

# Create the violin plot
sns.violinplot(x="Step", y="mse", hue="Category", data=df_combined, cut=0, palette={"Adaptive Regression": "blue", "Regression": "green", "ARIMA": "red"})

# Add title and labels
plt.title('MSE')
plt.xlabel('Step')
plt.ylabel('MSE')

# Create custom legend
#colors = ['blue', 'green', 'red']
#labels = ['Adaptive Regression', 'Regression', 'ARIMA']
#handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
sns.violinplot(x="Step", y="mae", hue="Category", data=df_combined, cut=0, palette={"Adaptive Regression": "blue", "Regression": "green", "ARIMA": "red"})

# Add title and labels
plt.title('MAE')
plt.xlabel('Step')
plt.ylabel('MAE')

plt.tight_layout()

plt.show()
