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
from sklearn.tree import DecisionTreeRegressor, plot_tree # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.inspection import PartialDependenceDisplay # type: ignore
import shap # type: ignore
from statsmodels import robust # type: ignore
import seaborn as sns # type: ignore
from scipy.interpolate import make_interp_spline # type: ignore

# Smoothing function
def smooth_curve(x, y, num_points=300):
    x_new = np.linspace(x.min(), x.max(), num_points)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_new)
    return x_new, y_smooth

n_clusters = 2 # Number of clusters

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
        'se': se,
        'mae': res[key]['mae'],
    }

    data[key] = features

# Convert to a DataFrame for easier manipulation
print(data)

# Convert to DataFrame
df = pd.DataFrame(data).T

# Plotting the violin plot
#plt.figure(figsize=(8, 6))
#sns.violinplot(data=df, y='mae', inner=None, color='lightgray')  # Violin plot
#sns.stripplot(data=df, y='mae', color='blue', size=10, jitter=True)  # Points inside
#plt.title('Violin Plot of Mean Absolute Error (MAE)')
#plt.ylabel('Mean Absolute Error (MAE)')
#plt.grid(True)
#plt.show()

# Assuming your data is in a DataFrame called df
correlation_matrix = df.corr()

mae_correlation = correlation_matrix['mae'].sort_values(ascending=False)
print(mae_correlation)

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)#
plt.title('Correlation Matrix')
plt.show()

"""
# Separate features and target
X = df.drop(columns=['mae'])
y = df['mae']

# Initialize and train the model
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)


# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print feature importances
print("Feature importances:")
for f in range(X.shape[1]):
    print(f"{X.columns[indices[f]]}: {importances[indices[f]]:.4f}")
"""
# Create partial dependence plots for each feature
"""
features = X.columns.tolist()
fig, ax = plt.subplots(len(features), 1, figsize=(10, 5 * len(features)))

for i, feature in enumerate(features):
    PartialDependenceDisplay.from_estimator(model, X, [feature], ax=ax[i])
    ax[i].set_title(f"Partial Dependence of MAE on {feature}")

plt.tight_layout()
plt.show()
"""
"""
# Initialize the SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Summary plot
shap.summary_plot(shap_values, X, plot_type="bar")
plt.show()
"""
"""
# Detailed SHAP plot for a single feature
shap.dependence_plot("mean", shap_values, X)
plt.show()
"""
# Features to remove
#features_to_remove = ['autocorr_lag2', 'se', 'mean', 'variance', 'skewness', 'quantile_50', 'range', 'min_val', 'kurtosis', 'std_dev', 'quantile_25', 'mad', 'medad']
#features_to_remove = ['autocorr_lag1', 'autocorr_lag2']
features_to_remove = []

X = df.drop(columns=features_to_remove)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create clusters
# Apply KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

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

# Add cluster labels to the DataFrame
df['cluster'] = clusters

# Step 3: Plotting violin plots of MAE values inside each cluster
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='cluster', y='mae', inner=None, color='lightgray')  # Violin plot
sns.stripplot(data=df, x='cluster', y='mae', color='blue', size=10, jitter=True)  # Points inside
plt.title('Violin Plot of Mean Absolute Error (MAE) by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Mean Absolute Error (MAE)')
plt.grid(True)
plt.show()

"""
# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
sns.scatterplot(data=df, x='max_val', y='mae', hue='cluster', palette='viridis', s=100)
plt.title('Clusters of Time Series Data')
plt.xlabel('Max Val')
plt.ylabel('MAE')
plt.legend(title='Cluster')
"""

# Pad the sequences so they have the same length
max_len = min(len(ts) for ts in time_series_values)
time_series_values_padded = pad_sequences(time_series_values, maxlen=max_len, padding='post', dtype='float')

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


# Plot histograms for the number of time series in each cluster
plt.subplot(2, 1, 2)
#plt.hist(clusters, bins=np.arange(n_clusters + 1) - 0.5, edgecolor='black', align='mid')
hist, bins, patches = plt.hist(clusters, bins=np.arange(n_clusters + 1) - 0.5, edgecolor='black', align='mid')

# Create a dictionary to store time series keys for each cluster
cluster_dict = {i: [] for i in range(n_clusters)}
for i, key in enumerate(time_series_keys):
    cluster_dict[clusters[i]].append(key)

# Add labels on top of each bar and add the cluster information to the legend
for i in range(n_clusters):
    plt.text(i, hist[i], str(round(float(avg_mae[i]), 2)), ha='center', va='bottom')

plt.xticks(range(n_clusters))
plt.xlabel('Cluster')
plt.ylabel('Number of Time Series')
plt.title('Number of Time Series in Each Cluster')

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
#time.sleep(5)

# Plot the average MAE and MSE for each cluster
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
for i in range(n_clusters):
    plt.plot(list(res_mae_avg.keys()), [item[i] for item in res_mae_avg.values()], label=f'Cluster {i}', color=colors[i], marker='x')

    x = np.arange(len([item[i] for item in res_mae_avg.values()]))
    x_smooth, mean_smooth = smooth_curve(x, [item[i] for item in res_mae_avg.values()])
    _, min_smooth = smooth_curve(x, [item[i] for item in res_mae_min.values()])
    _, max_smooth = smooth_curve(x, [item[i] for item in res_mae_max.values()])

    #plt.plot(x_smooth, mean_smooth, label=f'Cluster {i}', color=colors[i])
    plt.fill_between(
        x_smooth,
        min_smooth,
        max_smooth,
        color=colors[i],
        alpha=0.2  # Faded color for min-max range
    )

plt.title('Average MAE per Cluster')
plt.legend()

plt.subplot(2, 1, 2)
for i in range(n_clusters):
    plt.plot(list(res_mse_avg.keys()), [item[i] for item in res_mse_avg.values()], label=f'Cluster {i}', color=colors[i], marker='x')

    x = np.arange(len([item[i] for item in res_mse_avg.values()]))
    x_smooth, mean_smooth = smooth_curve(x, [item[i] for item in res_mse_avg.values()])
    _, min_smooth = smooth_curve(x, [item[i] for item in res_mse_min.values()])
    _, max_smooth = smooth_curve(x, [item[i] for item in res_mse_max.values()])

    #plt.plot(x_smooth, mean_smooth, label=f'Cluster {i}', color=colors[i])
    plt.fill_between(
        x_smooth,
        min_smooth,
        max_smooth,
        color=colors[i],
        alpha=0.2  # Faded color for min-max range
    )

plt.title('Average MSE per Cluster')
plt.legend()
plt.show()
