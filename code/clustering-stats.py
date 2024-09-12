import numpy as np # type: ignore
from sklearn.cluster import KMeans # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import json # type: ignore
from keras.preprocessing.sequence import pad_sequences # type: ignore
import time # type: ignore
from tslearn.clustering import TimeSeriesKMeans # type: ignore
from tslearn.preprocessing import TimeSeriesScalerMeanVariance # type: ignore
from scipy.spatial.distance import pdist, squareform # type: ignore
from statsmodels.tsa.stattools import acf # type: ignore
from scipy.stats import skew, kurtosis, iqr, variation, median_abs_deviation, sem # type: ignore
from statsmodels.tsa.stattools import acf # type: ignore
import pandas as pd # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.model_selection import LeaveOneOut # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error # type: ignore
from sklearn.tree import DecisionTreeRegressor, plot_tree # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.inspection import PartialDependenceDisplay # type: ignore
import shap # type: ignore
from statsmodels import robust # type: ignore
import seaborn as sns # type: ignore
from scipy.interpolate import make_interp_spline # type: ignore
from sklearn.feature_selection import RFE # type: ignore
import random # type: ignore
import sys

random.seed(1)

# Smoothing function
def smooth_curve(x, y, num_points=300):
    x_new = np.linspace(x.min(), x.max(), num_points)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_new)
    return x_new, y_smooth

n_clusters = 3 # Number of clusters

# Load the series_list from the file
with open('../data/series_list_customized_p.json', 'r') as file:
    series_list = json.load(file)

with open('../data/cluster_data.json', 'r') as file:
    regressor_perf = json.load(file) # َAdaptive Regressor Performance

#print(regressor_perf)
step = 1

perf_data = regressor_perf[str(step)]

res = {}
for elem in perf_data:  
    for key, value in elem.items():
        res[key] = {"mae": value['mae'], "mse": value['mse']}

print(res)
time.sleep(2)

time_series_data = series_list[0]

# List of keys to remove
keys_to_remove = [
    "m3-2_m3-2", "m3-3_m3-3", "m3-4_m3-4", "m3-5_m3-5",
    "m3-6_m3-6", "m3-7_m3-7", "m3-8_m3-8", "m3-9_m3-9",
    "m3-10_m3-10", "m3-11_m3-11", "m3-12_m3-12"
]

# Remove specified keys from the time series data
time_series_data_filtered = {k: v for k, v in time_series_data.items() if k not in keys_to_remove}

#print(time_series_data.values())

# Convert the dictionary to a list of time series
time_series_keys = list(time_series_data_filtered.keys())
time_series_values = list(time_series_data_filtered.values())

max_len = min(len(ts) for ts in time_series_values)

data = {}

for key in time_series_data_filtered.keys():
    time_series = time_series_data_filtered[key]

    # Statistical features
    mean = np.mean(time_series)
    std_dev = np.std(time_series)
    variance = np.var(time_series)
    skewness = skew(time_series)
    kurt = kurtosis(time_series)
    min_val = np.min(time_series)
    max_val = np.max(time_series)
    quantiles = np.percentile(time_series, [25, 50, 75])
    auto_corr = acf(time_series, nlags=10)

    # Calculating additional variability features
    range_val = np.max(time_series) - np.min(time_series)
    iqr_val = iqr(time_series)
    cv = variation(time_series)
    mad = robust.mad(time_series, axis=0)  # Mean Absolute Deviation
    medad = median_abs_deviation(time_series)
    rmd = np.mean(np.abs(time_series - np.mean(time_series))) / np.mean(time_series)
    se = sem(time_series)

    # Create a dictionary of features
    features = {
        'mean': mean,
        'std_dev': std_dev,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurt,
        'min_val': min_val,
        'max_val': max_val,
        'quantile_25': quantiles[0],
        'quantile_50': quantiles[1],
        'quantile_75': quantiles[2],
        'autocorr_lag1': auto_corr[1],
        'autocorr_lag2': auto_corr[2],
        'range': range_val,
        'iqr': iqr_val,
        'cv': cv,
        'mad': mad,
        'medad': medad,
        'rmd': rmd,
        'se': se
    }

    data[key] = features

# Convert to DataFrame
df = pd.DataFrame(data).T
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Create clusters
# Apply KMeans clustering
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(X_scaled)

df['cluster'] = clusters

# Create a dictionary to store time series keys for each cluster
cluster_dict = {i: [] for i in range(n_clusters)}
for i, key in enumerate(time_series_keys):
    cluster_dict[clusters[i]].append(key)

# Plot the original time series data for each cluster
plt.figure(figsize=(12, 8))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:gray']  # Colors for the clusters
labels = ['Good links', 'Bad links', 'Bursty links', 'Cluster 3', 'Cluster 4', 'Cluster 5']


# Plot histograms for the number of time series in each cluster
#plt.hist(clusters, bins=np.arange(n_clusters + 1) - 0.5, edgecolor='black', align='mid')
hist, bins, patches = plt.hist(clusters, bins=np.arange(n_clusters + 1) - 0.5, edgecolor='black', align='mid')

# Create a dictionary to store time series keys for each cluster
cluster_dict = {i: [] for i in range(n_clusters)}
for i, key in enumerate(time_series_keys):
    cluster_dict[clusters[i]].append(key)

plt.figure(figsize=(7, 6))
plt.rcParams.update({'font.size': 13})

mean_values = []
min_values = []
max_values = []

for i in range(n_clusters):
    cluster_keys = cluster_dict[i]

    cluster_data = [time_series_data_filtered[key][:max_len] for key in cluster_keys]

    print(cluster_data)
    
    # Calculate the mean, min, and max across the time series in the cluster
    mean_values = np.mean(cluster_data, axis=0)
    min_values = np.min(cluster_data, axis=0)
    max_values = np.max(cluster_data, axis=0)

    plt.plot(mean_values, label=f'Cluster {i} - {labels[i]} ({str(round(hist[i]))} links)' , color=colors[i])
    plt.fill_between(range(len(mean_values)), min_values, max_values, color=colors[i], alpha=0.2)
    
plt.xlabel('Time Window')
plt.ylabel('Number of received packets')
plt.legend()
plt.title('Mean Time Series for Each Cluster')
plt.show()
#plt.savefig("Clusters-Links.pdf", format="pdf", bbox_inches="tight")
#sys.exit(0)




res_mae_avg = {}
res_mse_avg = {}
res_mae_min = {}
res_mse_min = {}
res_mae_max = {}
res_mse_max = {}

for key, value in regressor_perf.items():
    step = key

    perf_data = regressor_perf[str(step)]

    res = {}
    for elem in perf_data:  
        for key, value in elem.items():
            res[key] = {"mae": value['mae'], "mse": value['mse']}

    avg_mae = {}
    avg_mse = {}
    min_mae = {}
    min_mse = {}
    max_mae = {}
    max_mse = {}
    # Print cluster assignments
    for i, key in enumerate(time_series_keys):
        #print(f"Time series {key} is in cluster {clusters[i]}")
        try:
            avg_mae[clusters[i]] = avg_mae.get(clusters[i], 0) + res[key]['mae']
            avg_mse[clusters[i]] = avg_mse.get(clusters[i], 0) + res[key]['mse']

            print("Cluster: ", clusters[i], " MAE: ", res[key]['mae'], " MSE: ", res[key]['mse'])

            min_mae[clusters[i]] = min(min_mae.get(clusters[i], 0), res[key]['mae'])
            max_mae[clusters[i]] = max(max_mae.get(clusters[i], 0), res[key]['mae'])
            min_mse[clusters[i]] = min(min_mse.get(clusters[i], 0), res[key]['mse'])
            max_mse[clusters[i]] = max(max_mse.get(clusters[i], 0), res[key]['mse'])

        except Exception as e:
            print("here:", e)
            time.sleep(4)
            pass

    for i in range(n_clusters):
        avg_mae[i] /= len(clusters[clusters == i])
        avg_mse[i] /= len(clusters[clusters == i])

    print("Avg MAE per cluster for step: ", step, " is: ", avg_mae)
    print("Avg MSE per cluster: for step: ", step, " is: ", avg_mse)

    res_mae_avg[step] = avg_mae
    res_mse_avg[step] = avg_mse
    res_mae_min[step] = min_mae
    res_mse_min[step] = min_mse
    res_mae_max[step] = max_mae
    res_mse_max[step] = max_mse

print(res_mae_avg, res_mae_min, res_mae_max)
#time.sleep(5)
# Plot the average MAE and MSE for each cluster
"""
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
for i in range(n_clusters):
    plt.plot(list(res_mae_avg.keys()), [item[i] for item in res_mae_avg.values()], label=f'Cluster {i}', color=colors[i], marker='x')

    x = np.arange(len([item[i] for item in res_mae_avg.values()]))
    x_smooth, mean_smooth = smooth_curve(x, [item[i] for item in res_mae_avg.values()])
    _, min_smooth = smooth_curve(x, [item[i] for item in res_mae_min.values()])
    _, max_smooth = smooth_curve(x, [item[i] for item in res_mae_max.values()])

    #plt.plot(x_smooth, mean_smooth, label=f'Cluster {i}', color=colors[i])
    #plt.fill_between(
    #    x_smooth,
    #    min_smooth,
    #    max_smooth,
    #    color=colors[i],
    #    alpha=0.2  # Faded color for min-max range
    #)
"""


step = 1

perf_data = regressor_perf[str(step)]

res = {}
for elem in perf_data:  
    for key, value in elem.items():
        res[key] = {"mae": value['mae'], "mse": value['mse']}

# Create a dictionary to store time series values for each cluster
# Prepare the data for the violin plot
violin_data = []
violin_labels = []

for cluster in range(n_clusters):
    for i, key in enumerate(time_series_keys):
        if clusters[i] == cluster:
            violin_data.append(res[key]["mae"])
            violin_labels.append(cluster)

            if res[key]["mae"] > 5:
                print(key, res[key]["mae"])
                time.sleep(5)

# Prepare the figure for all violin plots in one plot
#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))

plt.figure(figsize=(7, 6))
plt.rcParams.update({'font.size': 13})


# Prepare data for the combined MAE violin plot
all_mae_violin_data = []
all_mae_violin_labels = []
all_mae_clusters = []

# Prepare data for the combined MSE violin plot
all_mse_violin_data = []
all_mse_violin_labels = []
all_mse_clusters = []

for step in range(1, 11):
    if str(step) in regressor_perf:
        perf_data = regressor_perf[str(step)]

        res = {}
        for elem in perf_data:  
            for key, value in elem.items():
                res[key] = {"mae": value['mae'], "mse": value['mse']}

        for cluster in range(n_clusters):
            for i, key in enumerate(time_series_keys):
                if clusters[i] == cluster:
                    all_mae_violin_data.append(res[key]["mae"])
                    all_mae_violin_labels.append(f'Step {step}')
                    all_mae_clusters.append(f'Cluster {cluster}')
                    
                    all_mse_violin_data.append(res[key]["mse"])
                    all_mse_violin_labels.append(step)
                    all_mse_clusters.append(f'Cluster {cluster} - {labels[cluster]}')

all_rmse_violin_data = np.sqrt(all_mse_violin_data)


"""
# Create the violin plot for MAE
sns.boxplot(ax=ax1, x=all_mae_violin_labels, y=all_mae_violin_data, hue=all_mae_clusters, 
               palette=colors[:n_clusters])

# Add title and labels for MAE plot
ax1.set_title('MAE for Each Step and Cluster')
ax1.set_xlabel('Step')
ax1.set_ylabel('MAE')
#ax1.set_yscale('log')
# Add legend for MAE plot
handles, _ = ax1.get_legend_handles_labels()
ax1.legend(handles, [f'Cluster {i}' for i in range(n_clusters)], title='Cluster')
ax1.grid()
# Create the violin plot for MSE
"""
sns.boxplot(x=all_mse_violin_labels, y=all_rmse_violin_data, hue=all_mse_clusters, 
               palette=colors[:n_clusters])

# Add title and labels for MSE plot
plt.title('RMSE')
plt.xlabel('Prediction Step')
plt.ylabel('RMSE')
#ax2.set_yscale('log')

# Add legend for MSE plot
#handles, _ = plt.get_legend_handles_labels()
plt.legend()
plt.grid()

#plt.xticks(rotation=90)
plt.tight_layout()
plt.yscale('log')
#plt.show()
plt.savefig("Clusters-Metrics.pdf", format="pdf", bbox_inches="tight")
sys.exit(0)