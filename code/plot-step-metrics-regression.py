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
    # Function to parse the data file
    def parse_data(file_path):
        data = []
        current_step = None

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("Step:"):
                    current_step = int(line.split()[1])
                    #print(current_step)
                elif line.startswith("False Positive"):
                    metrics = line.split()
                    false_positive = int(metrics[2])
                    false_negative = int(metrics[5])
                    true_positive = int(metrics[8])
                    true_negative = int(metrics[11])
                    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
                    recall = true_positive / (true_positive + false_negative)
                    specificity = true_negative / (true_negative + false_positive)
                    precision = true_positive / (true_positive + false_positive)
                    data.append({
                        "Step": current_step,
                        "Accuracy": accuracy,
                        "Recall": recall,
                        "Specificity": specificity,
                        "Precision": precision
                })
                    
        return pd.DataFrame(data)
                    
    
    # Parse the data from the file
    result = []
    #filenames = ['../data/result-regression-1.json', '../data/result-regression-2.json', '../data/result-regression-3.json', '../data/result-regression-4.json']
    """
    filenames = ['../data/regression-1.json', '../data/regression-2.json', '../data/regression-3.json', '../data/regression-4.json']
    for filename in filenames:
        with open(filename, 'r') as file:
            series_list = json.load(file)
            result.extend(series_list)
    """
    #with open('../data/result-complete-regression.json', 'w') as file:
    #    json.dump(result, file)
    # Ensure 'Step' is treated as a categorical variable
    #print(result)

    filename = '../data/results-regression-2.json'
    with open(filename, 'r') as file:
        result = json.load(file)

    accuracy = 0
    recall = 0
    specificity = 0
    precision = 0

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
                    #print("here ", key_2, value_2)
                    #time.sleep(1)
                    for elem in value_2:
                        for key_3, value_3 in elem.items():
                            #print(key_3, value_3["true_positive"], key)
                            step = key
                            data_ = data_expe[key_3]
                            size = int(len(data) * 0.66) # First 2/3 of the data are used for training, and the rest is used for testing
                            test = data_[size:len(data_)]

                            y_pred = value_3["predictions"]
                            test = test[:len(y_pred)]

                            #print(key_3, y_pred, test, len(test))
                            """
                            true_positive = 0
                            true_negative = 0
                            false_positive = 0
                            false_negative = 0

                            for i in range(len(y_pred)):
                                if test[i] <= SLA and y_pred[i] <= SLA:
                                    true_positive += 1
                                elif test[i] >= SLA and y_pred[i] >= SLA:
                                    true_negative += 1
                                elif test[i] > SLA and y_pred[i] < SLA:
                                    false_positive += 1
                                    #print("here, ", i, y[i], y_pred[i])
                                    #time.sleep(1)
                                elif test[i] < SLA and y_pred[i] > SLA:
                                    false_negative += 1

                            accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)
                            try:
                                recall = true_positive  / (true_positive + false_negative)
                                specificity = true_negative / (true_negative + false_positive)
                                precision = true_positive / (true_positive + false_positive)
                            except ZeroDivisionError:
                                if (true_positive + false_negative) == 0:
                                    recall = 0
                                if (true_negative + false_positive) == 0:
                                    specificity = 0
                                if (true_positive + false_positive) == 0:
                                    precision = 0
                            
                            mae = 0
                            mae = mean_absolute_error(test, y_pred)
                            mse = mean_squared_error(test, y_pred)
                            """
                            mae = value_3["MAE"]
                            mse = value_3["MSE"]
                            print("MAE: ", mae, " MSE: ", mse, " step: ", step)
                            if mse > 300:
                                print("HERE MAE: ", mae, " MSE: ", mse, " step: ", step)
                                time.sleep(5)
                            """
                            accuracy = (value_3["true_positive"] + value_3["true_negative"]) / (value_3["true_positive"] + value_3["true_negative"] + value_3["false_positive"] + value_3["false_negative"])
                            recall = value_3["true_positive"] / (value_3["true_positive"] + value_3["false_negative"])
                            try:
                                specificity = value_3["true_negative"] / (value_3["true_negative"] + value_3["false_positive"])
                                precision = value_3["true_positive"] / (value_3["true_positive"] + value_3["false_positive"])
                            except ZeroDivisionError:
                                specificity = 0
                                precision = 0
                            """
                            data.append({
                                "Step": int(step),
                                "Accuracy": float(accuracy),
                                "Recall": float(recall),
                                "Specificity": float(specificity),
                                "Precision": float(precision),
                                "mae": mae,
                                "mse": mse
                            })

                            #predictions = value_3["predictions"][:len(data_expe[key_3])]

                            #size = int(len(data_expe[key_3]) * 0.66) # First 2/3 of the data are used for training, and the rest is used for testing

                            # Convert ground truth values to binary classes based on the threshold
                            #y_true = np.array([1 if val >= threshold else 0 for val in data_expe[key_3][size:len(data_expe[key_3])]])

                            #print("y_true: ", y_true, "predictions: ", predictions)
                            #time.sleep(1)
                            # Use predicted values as scores
                            #y_scores = np.array(predictions)

                            #all_ground_truths.extend(y_true)
                            #all_predictions.extend(y_scores)
                except Exception as e:
                    print("Exception: ", e)
                    #print(all_predictions)
                    pass
                        # Convert lists to numpy arrays
        """
        all_ground_truths = np.array(all_ground_truths[:len(all_predictions)])
        all_predictions = np.array(all_predictions)

        # Applying softmax along axis 1
        print(all_predictions)
        probabilities = np.array(softmax(all_predictions))
        print(probabilities)

        #time.sleep(1)

        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(all_ground_truths, probabilities)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Step {step}')
        if cpt == 0:
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            cpt += 1

        # Print TPR, FPR, and thresholds for reference
        print(f"False Positive Rate: {fpr}")
        print(f"True Positive Rate: {tpr}")
        print(f"Thresholds: {thresholds}")

        # Plot ROC curve
        
    #plt.figure()
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
        
    """
    
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by="Step")
    print(data)
    mean_accuracy_step1 = df_sorted[df_sorted["Step"] == 1]["Accuracy"].mean()
    print("Mean Accuracy for Step 1: ", mean_accuracy_step1)
    print("Mean Accuracy: ", df_sorted["Accuracy"].mean())
    print("Mean Recall: ", df_sorted["Recall"].mean())
    print("Mean Specificity: ", df_sorted["Specificity"].mean())
    print("Mean Precision: ", df_sorted["Precision"].mean())
    print("Mean MAE: ", df_sorted["mae"].mean())

    print(df_sorted.head())
    # Create violin plots for accuracy and recall
    plt.figure(figsize=(12, 6))

    # MAE
    #plt.subplot(2, 2, 1)
    sns.violinplot(x="Step", y="mse", data=df_sorted)
    plt.title('MSE')
    plt.show()

    sns.violinplot(x="Step", y="mae", data=df_sorted)
    plt.title('MAE')
    plt.show()
    
    # Accuracy
    plt.subplot(2, 2, 1)
    sns.violinplot(x="Step", y="Accuracy", data=df_sorted)
    plt.title('Accuracy')

    # Recall
    plt.subplot(2, 2, 2)
    sns.violinplot(x="Step", y="Recall", data=df_sorted)
    plt.title('Recall')

    plt.subplot(2, 2, 3)
    sns.violinplot(x="Step", y="Specificity", data=df)
    plt.title('Specificity')

    plt.subplot(2, 2, 4)
    sns.violinplot(x="Step", y="Precision", data=df)
    plt.title('Precision')
  
    plt.tight_layout()
    plt.show()