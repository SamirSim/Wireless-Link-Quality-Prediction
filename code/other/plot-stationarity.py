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
from statsmodels.tsa.stattools import adfuller, kpss # type: ignore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # type: ignore
from statsmodels.tsa.seasonal import seasonal_decompose # type: ignore

def adf_test(timeseries):
    result = adfuller(timeseries)
    return result[1]  # p-value

def kpss_test(timeseries):
    result = kpss(timeseries)
    return result[1]  # p-value

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
plt.rcParams.update({'font.size': 12})

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

adf_results = []
kpss_results = []

for i in range(n_clusters):
    cluster_keys = cluster_dict[i]

    cluster_data = [time_series_data_filtered[key][:max_len] for key in cluster_keys]

    mean_values = np.mean(cluster_data, axis=0)

    #plt.figure(figsize=(12, 6))
    #for series in cluster_data[:10]:  # Decompose a few sample series
    decomposition = seasonal_decompose(mean_values, model='additive', period=3)
    decomposition.plot()
    plt.title(f'Decomposition of Time Series in Cluster {i}')
    plt.show()

# Initialize arrays to store the metrics for each cluster
means_all = []
variances_all = []
skewnesses_all = []
kurtoses_all = []

# Calculate metrics for each cluster
for i in range(n_clusters):
    cluster_keys = cluster_dict[i]

    cluster_data = [time_series_data_filtered[key][:max_len] for key in cluster_keys]
    
    means = [np.mean(ts) for ts in cluster_data]
    variances = [np.var(ts) for ts in cluster_data]
    skewnesses = [skew(ts) for ts in cluster_data]
    kurtoses = [kurtosis(ts) for ts in cluster_data]

    means_all.append(means)
    variances_all.append(variances)
    skewnesses_all.append(skewnesses)
    kurtoses_all.append(kurtoses)

    print(means, variances, skewnesses, kurtoses)

# Create subplots for each metric
fig, axs = plt.subplots(2, 2, figsize=(8, 5))
plt.grid()
# Boxplot of means
axs[0, 0].boxplot(means_all)
axs[0, 0].set_title('Mean')
axs[0, 0].set_ylabel('Mean')
axs[0, 0].set_xticklabels([f'Cluster {i+1}' for i in range(n_clusters)])
axs[0, 0].grid(True)  # Add grid

# Boxplot of variances
axs[0, 1].boxplot(variances_all)
axs[0, 1].set_title('Variance')
axs[0, 1].set_ylabel('Variance')
axs[0, 1].set_xticklabels([f'Cluster {i+1}' for i in range(n_clusters)])
axs[0, 1].grid(True)  # Add grid

# Boxplot of skewnesses
axs[1, 0].boxplot(skewnesses_all)
axs[1, 0].set_title('Skewness')
axs[1, 0].set_ylabel('Skewness')
axs[1, 0].set_xticklabels([f'Cluster {i+1}' for i in range(n_clusters)])
axs[1, 0].grid(True)  # Add grid

axs[1, 1].boxplot(kurtoses_all)
axs[1, 1].set_title('Kurtosis')
axs[1, 1].set_ylabel('Kurtosis')
axs[1, 1].set_xticklabels([f'Cluster {i+1}' for i in range(n_clusters)])
axs[1, 1].grid(True)  # Add grid

# Adjust layout and display the figure
plt.tight_layout()
#plt.show()

plt.savefig("Clusters-Stats.pdf", format="pdf", bbox_inches="tight")
sys.exit(0)