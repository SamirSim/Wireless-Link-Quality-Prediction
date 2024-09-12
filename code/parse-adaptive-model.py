import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
import json
import time
from sklearn.metrics import roc_curve, auc # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc # type: ignore
import numpy as np # type: ignore
from scipy.special import softmax # type: ignore

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
                            time.sleep(5)
                        mse_list.append(mse)
                        maes_list.append(mae)
                        print("Final MSE: ", mse, "Final MAE: ", mae)

                except Exception as e:
                    print("Exception: ", e)
                    print(step, key_2, key_3)
                    time.sleep(10)
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


df = pd.DataFrame(data)
df_sorted = df.sort_values(by="Step")

"""
print(data)
mean_accuracy_step1 = df_sorted[df_sorted["Step"] == 1]["Accuracy"].mean()
#(len(df_sorted[df_sorted["Step"] == 10]["Accuracy"]))

print("Mean MAE: ", df_sorted["mae"].mean())
"""

print("Mean MAE: ", df_sorted["mae"].mean())
print("Mean MSE: ", df_sorted["mse"].mean())
print("Number of Step 1 elements: ", len(df_sorted[df_sorted["Step"] == 2]))
print("Number of Step 10 elements: ", len(df_sorted[df_sorted["Step"] == 10]["mae"]))

print(df_sorted)
print(df_sorted.head())

# Create violin plots for accuracy and recall
plt.figure(figsize=(12, 6))
print(np.array(df_sorted[df_sorted["Step"] == 20]["mse"]), len(df_sorted[df_sorted["Step"] == 20]["mse"]))

plt.subplot(2, 1, 1)
sns.violinplot(x="Step", y="mse", data=df_sorted, cut=0)
plt.title('Adaptive Regression')

plt.subplot(2, 1, 2)
sns.violinplot(x="Step", y="mae", data=df_sorted, cut=0)

plt.show()
