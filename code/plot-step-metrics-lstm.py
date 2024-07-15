import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
import json
import time
from sklearn.metrics import roc_curve, auc # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc # type: ignore
import numpy as np # type: ignore
from scipy.special import softmax # type: ignore

threshold = 40
SLA = threshold

MEAN = False
if MEAN: # Plot mean values for the metrics
    data_str = """
    Step:  3
    False Positive:  14  False Negative:  27  True Positive:  38  True Negative:  20
    Step:  3
    False Positive:  12  False Negative:  6  True Positive:  79  True Negative:  2
    Step:  3
    False Positive:  15  False Negative:  3  True Positive:  27  True Negative:  0
    Step:  3
    False Positive:  8  False Negative:  3  True Positive:  33  True Negative:  1
    Step:  3
    False Positive:  0  False Negative:  0  True Positive:  84  True Negative:  0
    Step:  3
    False Positive:  1  False Negative:  1  True Positive:  82  True Negative:  0
    Step:  3
    False Positive:  0  False Negative:  0  True Positive:  60  True Negative:  0
    Step:  3
    False Positive:  0  False Negative:  6  True Positive:  54  True Negative:  0
    Step:  3
    False Positive:  16  False Negative:  1  True Positive:  73  True Negative:  0
    Step:  3
    False Positive:  26  False Negative:  10  True Positive:  42  True Negative:  12
    Step:  3
    False Positive:  40  False Negative:  6  True Positive:  66  True Negative:  2
    Step:  3
    False Positive:  19  False Negative:  22  True Positive:  64  True Negative:  9
    Step:  3
    False Positive:  17  False Negative:  24  True Positive:  12  True Negative:  13
    Step:  3
    False Positive:  9  False Negative:  27  True Positive:  16  True Negative:  14
    Step:  3
    False Positive:  0  False Negative:  0  True Positive:  66  True Negative:  0
    Step:  3
    False Positive:  7  False Negative:  0  True Positive:  59  True Negative:  0

    """

    # Parse the data
    lines = data_str.strip().split("\n")
    data = []

    for i in range(0, len(lines), 2):
        metrics = lines[i+1].split()
        false_positive = int(metrics[2])
        false_negative = int(metrics[5])
        true_positive = int(metrics[8])
        true_negative = int(metrics[11])

        data.append({
            "False Positive": false_positive,
            "False Negative": false_negative,
            "True Positive": true_positive,
            "True Negative": true_negative
        })

    # Calculate the mean values
    #mean_accuracy = sum(item["True Positive"] + item["True Negative"] for item in data) / sum(item["False Positive"] + item["True Negative"] + item["False Negative"] + item["True Positive"] for item in data)
    mean_false_positive = sum(item["False Positive"] for item in data) / len(data) / sum(item["False Positive"] + item["True Negative"] + item["False Negative"] + item["True Positive"] for item in data)
    mean_false_negative = sum(item["False Negative"] for item in data) / len(data) / sum(item["False Positive"] + item["True Negative"] + item["False Negative"] + item["True Positive"] for item in data)
    mean_true_positive = sum(item["True Positive"] for item in data) / len(data) / sum(item["False Positive"] + item["True Negative"] + item["False Negative"] + item["True Positive"] for item in data)
    mean_true_negative = sum(item["True Negative"] for item in data) / len(data) / sum(item["False Positive"] + item["True Negative"] + item["False Negative"] + item["True Positive"] for item in data)
    mean_accuracy = (mean_true_positive + mean_true_negative) / (mean_false_positive + mean_true_negative + mean_false_negative + mean_true_positive)
    mean_misclassification_rate = (mean_false_positive + mean_false_negative) / (mean_false_positive + mean_true_negative + mean_false_negative + mean_true_positive)
    mean_recall = mean_true_positive / (mean_true_positive + mean_false_negative)
    mean_specificity = mean_true_negative / (mean_true_negative + mean_false_positive)
    mean_precision = mean_true_positive / (mean_true_positive + mean_false_positive)

    print(mean_accuracy, mean_misclassification_rate, mean_recall, mean_specificity, mean_precision)

    # Define the data
    steps = list(range(1, 11))
    mean_false_positive = [11.375, 10.9375, 11.5, 10.8125, 10.875, 10.8125, 10.5, 10.3125, 10.5625, 9.875]
    mean_false_negative = [8.125, 8.8125, 8.5, 9.5, 8.125, 9.1875, 10.625, 10.9375, 9.125, 8.6875]
    mean_true_positive = [54.8125, 53.75, 53.4375, 53.0, 53.3125, 52.3125, 49.8125, 48.8125, 50.5, 50.6875]
    mean_true_negative = [4.8125, 5.2, 4.5625, 5.1875, 5.1875, 4.9375, 5.1875, 4.9375, 5.1875, 5.75]

    mean_accuracy = [(mean_true_positive[i] + mean_true_negative[i]) / (mean_false_positive[i] + mean_true_negative[i] + mean_false_negative[i] + mean_true_positive[i]) for i in range(len(steps))]
    mean_misclassification_rate = [(mean_false_positive[i] + mean_false_negative[i]) / (mean_false_positive[i] + mean_true_negative[i] + mean_false_negative[i] + mean_true_positive[i]) for i in range(len(steps))]
    mean_recall = [mean_true_positive[i] / (mean_true_positive[i] + mean_false_negative[i]) for i in range(len(steps))] 
    mean_specificity = [mean_true_negative[i] / (mean_true_negative[i] + mean_false_positive[i]) for i in range(len(steps))]
    mean_precision = [mean_true_positive[i] / (mean_true_positive[i] + mean_false_positive[i]) for i in range(len(steps))]

    # Plot the data
    plt.figure(figsize=(12, 8))

    #plt.plot(steps, mean_false_positive, label='Mean False Positive', marker='o')
    #plt.plot(steps, mean_false_negative, label='Mean False Negative', marker='o')
    #plt.plot(steps, mean_true_positive, label='Mean True Positive', marker='o')
    #plt.plot(steps, mean_true_negative, label='Mean True Negative', marker='o')
    plt.plot(steps, mean_accuracy, label='Mean Accuracy', marker='o')
    #plt.plot(steps, mean_misclassification_rate, label='Mean Misclassification Rate', marker='o')
    plt.plot(steps, mean_recall, label='Mean Recall', marker='o')
    plt.plot(steps, mean_specificity, label='Mean Specificity', marker='o')
    plt.plot(steps, mean_precision, label='Mean Precision', marker='o')

    plt.xlabel('Steps')
    plt.ylabel('Mean Values')
    plt.title('Evolution of Mean Metrics Over Steps')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()


    """
    # Step 1
    (11.375 8.125 54.8125 4.8125)

    # Step 2
    (10.9375 8.8125 53.75 5.2)

    # Step 3
    (11.5 8.5 53.4375 4.5625)

    # Step 4
    (10.8125 9.5 53.0 5.1875)

    # Step 5
    (10.875 8.125 53.3125 5.1875)

    # Step 6
    (10.8125, 9.1875, 52.3125, 4.9375)

    # Step 7
    (10.5, 10.625, 49.8125, 5.1875)

    # Step 8
    (10.3125 10.9375 48.8125 4.9375)

    # Step 9
    (10.5625, 9.125, 50.5, 5.1875)

    # Step 10
    (9.875 8.6875 50.6875 5.75)

    """
else:
    # Parse the data from the file
    result = []
   
    # Load the series_list from the file
    with open('../data/series_list_customized_p.json', 'r') as file:
        series_list = json.load(file)

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
    
    df_rnn = pd.DataFrame(data)
    df_sorted_rnn = df_rnn.sort_values(by="Step")

    # LSTM
    filename = '../data/lstm.json'
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
    
    df_lstm = pd.DataFrame(data)
    df_sorted_lstm = df_lstm.sort_values(by="Step")

    # GRU
    filename = '../data/gru.json'
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
    
    df_gru= pd.DataFrame(data)
    df_sorted_gru = df_gru.sort_values(by="Step")

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    # Plot the violin plots for each dataset
    # Add a category column to each dataframe
    df_sorted_lstm['Category'] = 'LSTM'
    df_sorted_rnn['Category'] = 'RNN'
    df_sorted_gru['Category'] = 'GRU'

    # Concatenate the dataframes
    df_combined = pd.concat([df_sorted_rnn, df_sorted_lstm, df_sorted_gru])

    # Create the violin plot
    #sns.violinplot(x="Step", y="mse", hue="Category", data=df_combined, cut=0, palette={"Adaptive Regression": "blue", "Regression": "green"})
    #sns.violinplot(x="Step", y="mse", hue="Category", data=df_combined, palette={"LSTM": "blue", "RNN": "green", "GRU": "red"})
    sns.boxplot(x="Step", y="mse", hue="Category", data=df_combined, palette={"LSTM": "blue", "RNN": "green", "GRU": "red"})
    #plt.yscale('log')

    # Add title and labels
    plt.title('MSE')
    plt.xlabel('Step')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    #sns.violinplot(x="Step", y="mae", hue="Category", data=df_combined, cut=0, palette={"Adaptive Regression": "blue", "Regression": "green"})
    #sns.violinplot(x="Step", y="mae", hue="Category", data=df_combined, palette={"LSTM": "blue", "RNN": "green", "GRU": "red"})
    sns.boxplot(x="Step", y="mae", hue="Category", data=df_combined, palette={"LSTM": "blue", "RNN": "green", "GRU": "red"})
    #plt.yscale('log')

    # Add title and labels
    plt.title('MAE')
    plt.xlabel('Step')
    plt.ylabel('MAE')
    plt.grid()

    plt.tight_layout()

    plt.show()