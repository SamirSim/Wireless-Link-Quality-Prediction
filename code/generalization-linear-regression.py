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

#print(regressor_perf)
step = 1

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
#print(data)

# Convert to DataFrame
df = pd.DataFrame(data).T

# Separate features and target
X = df.drop(columns=['mae'])
y = df['mae']

# Initialize and train the model
model = LinearRegression()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

selector = RFE(model, n_features_to_select=5)  # Select top 5 features
selector = selector.fit(X_scaled, y)

# Get the selected features
selected_features = X.columns[selector.support_]
print('Selected features:', selected_features)

# Fit the model using selected features
X_selected = X[selected_features]

# Initialize Leave-One-Out Cross-Validation
loo = LeaveOneOut()
errors = []

X = X_selected

# Loop through each train-test split
for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    model.fit(X_train, y_train)
    
    # Predict the left-out sample
    y_pred = model.predict(X_test)

    #print(f'Predicted: {y_pred[0]:.4f}, Actual: {y_test.iloc[0]:.4f}', f'Error: {abs(y_pred[0] - y_test.iloc[0]):.4f}')
    
    # Compute the error
    error = mean_absolute_error(y_test, y_pred)

    if error > 100:
        print(X_train, X_test, y_train, y_test)
        time.sleep(2)
    error_mse = mean_squared_error(y_test, y_pred)
    errors.append(error)

# Compute average prediction error
average_error = sum(errors) / len(errors)
print(f'Average Prediction Error (MAE): {average_error:.4f}')
print(f'Mean Squared Error (MSE): {error_mse:.4f}')


# Convert to DataFrame
df = pd.DataFrame(data).T
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Create clusters
# Apply KMeans clustering
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(X_scaled)

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

# Step 4: Calculate average prediction error for each cluster
cluster_errors = []

for cluster in range(n_clusters):
    # Step 3: Initialize model
    model = LinearRegression()
    # Extract data for the current cluster
    cluster_data = df[df['cluster'] == cluster]
    X_cluster = cluster_data.drop(columns=['mae', 'cluster'])
    y_cluster = cluster_data['mae']
    
    X_cluster = X_cluster[selected_features]

    # Initialize Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    errors = []

    # Loop through each train-test split within the cluster
    for train_index, test_index in loo.split(X_cluster):
        X_train, X_test = X_cluster.iloc[train_index], X_cluster.iloc[test_index]
        y_train, y_test = y_cluster.iloc[train_index], y_cluster.iloc[test_index]

        # Train the model
        model.fit(X_train, y_train)

        # Predict the left-out sample
        y_pred = model.predict(X_test)

        # Compute the error
        error = mean_absolute_error(y_test, y_pred)

        if error > 100:
            print(X_train, X_test, y_train, y_test)
            time.sleep(2)
            
        else:
            errors.append(error)

        #print(f'Cluster {cluster}: Predicted: {y_pred[0]:.4f}, Actual: {y_test.iloc[0]:.4f}', f'Error: {abs(y_pred[0] - y_test.iloc[0]):.4f}')

    # Compute average prediction error for the current cluster
    average_error = sum(errors) / len(errors)
    cluster_errors.append((cluster, average_error))

# Print overall cluster errors
for cluster, error in cluster_errors:
    print(f'Cluster {cluster}: Average Prediction Error (MAE): {error:.4f}')