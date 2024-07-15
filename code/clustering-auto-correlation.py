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

n_clusters = 4 # Number of clusters

# Load the series_list from the file
with open('../data/series_list_customized_p.json', 'r') as file:
    series_list = json.load(file)

with open('../data/cluster_data.json', 'r') as file:
    regressor_perf = json.load(file) # َAdaptive Regressor Performance

print(regressor_perf)
step = 2

perf_data = regressor_perf[str(step)]

res = {}
for elem in perf_data:  
    for key, value in elem.items():
        res[key] = {"mae": value['mae'], "mse": value['mse']}

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

# Pad the sequences so they have the same length
max_len = min(len(ts) for ts in time_series_values)
time_series_values_padded = pad_sequences(time_series_values, maxlen=max_len, padding='post', dtype='float')

# Function to calculate autocorrelation
def calculate_acf(series, nlags):
    return acf(series, nlags=nlags, fft=True)

# Calculate ACF for each time series
nlags = 50  # Number of lags to include in the autocorrelation
acf_values = np.array([calculate_acf(ts, nlags) for ts in time_series_values_padded])

# Compute the pairwise distance matrix using the Euclidean distance between ACFs
distance_matrix = squareform(pdist(acf_values, metric='euclidean'))
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

clusters = kmeans.fit_predict(distance_matrix)

avg_mae = {}
avg_mse = {}
# Print cluster assignments
for i, key in enumerate(time_series_keys):
    #print(f"Time series {key} is in cluster {clusters[i]}")
    try:
        avg_mae[clusters[i]] = avg_mae.get(clusters[i], 0) + res[key]['mae']
        avg_mse[clusters[i]] = avg_mse.get(clusters[i], 0) + res[key]['mse']
    except:
        pass

for i in range(n_clusters):
    avg_mae[i] /= len(clusters[clusters == i])
    avg_mse[i] /= len(clusters[clusters == i])

print("Avg MAE per cluster: ", avg_mae)
print("Avg MSE per cluster: ", avg_mse)

# Optional: Plot the clusters
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)

colors = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
    'tab:purple', 'tab:gray', 'tab:brown', 'tab:pink', 
    'tab:olive', 'tab:cyan'
]

for i in range(n_clusters):
    cluster_indices = clusters == i
    cluster_values = time_series_values_padded[cluster_indices]

    mean_values = np.mean(cluster_values, axis=0)
    min_values = np.min(cluster_values, axis=0)
    max_values = np.max(cluster_values, axis=0)

    plt.plot(mean_values, label=f'Cluster {i}', color=colors[i])
    plt.fill_between(
        range(len(mean_values)),
        min_values,
        max_values,
        color=colors[i],
        alpha=0.2  # Faded color for min-max range
    )

plt.legend()
plt.title('Clustered Time Series')
plt.xlabel('Time Point')
plt.ylabel('Value')
#plt.show()

# Plot histograms for the number of time series in each cluster
plt.subplot(2, 1, 2)
#plt.hist(clusters, bins=np.arange(n_clusters + 1) - 0.5, edgecolor='black', align='mid')
hist, bins, patches = plt.hist(clusters, bins=np.arange(n_clusters + 1) - 0.5, edgecolor='black', align='mid')

# Create a dictionary to store time series keys for each cluster
cluster_dict = {i: [] for i in range(n_clusters)}
for i, key in enumerate(time_series_keys):
    cluster_dict[clusters[i]].append(key)

# Add labels on top of each bar and add the cluster information to the legend
#for i in range(n_clusters):
    #plt.text(i, hist[i], str(float(avg_mae[i])), ha='center', va='bottom')

plt.xticks(range(n_clusters))
plt.xlabel('Cluster')
plt.ylabel('Number of Time Series')
plt.title('Number of Time Series in Each Cluster with Avg MAE for step ' + str(step))

plt.tight_layout()
plt.show()

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
time.sleep(5)
# Plot the average MAE and MSE for each cluster
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
for i in range(n_clusters):
    plt.plot(list(res_mae_avg.keys()), [item[i] for item in res_mae_avg.values()], label=f'Cluster {i}', color=colors[i], marker='x')

    plt.fill_between(
        list(res_mae_avg.keys()),
        [item[i] for item in res_mae_min.values()],
        [item[i] for item in res_mae_max.values()],
        color=colors[i],
        alpha=0.2  # Faded color for min-max range
    )

plt.title('Average MAE per Cluster')
plt.legend()

plt.subplot(2, 1, 2)
for i in range(n_clusters):
    plt.plot(list(res_mse_avg.keys()), [item[i] for item in res_mse_avg.values()], label=f'Cluster {i}', color=colors[i], marker='x')

    plt.fill_between(
        list(res_mse_avg.keys()),
        [item[i] for item in res_mse_min.values()],
        [item[i] for item in res_mse_max.values()],
        color=colors[i],
        alpha=0.2  # Faded color for min-max range
    )

plt.title('Average MSE per Cluster')
plt.legend()
plt.show()
