import json
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
from statsmodels.tsa.stattools import adfuller # type: ignore
import pandas as pd # type: ignore

# Load JSON files (replace with actual file paths)
with open("../data/best-model-continuous-24h.json", "r") as f:
    fixed_data = json.load(f)

with open("../data/adaptive-model-continuous-24h.json", "r") as f:
    adaptive_data = json.load(f)

# Load JSON files (replace with actual file paths)
with open("../data/series-iotj-24h.json", "r") as f:
    data_expe = json.load(f)

# Extract data for boxplots
steps = sorted({step for link in fixed_data for step in fixed_data[link]})  # Get all unique steps

def classify_series(time_series):
    mean_value = np.mean(time_series)

    if 0 < mean_value < 9:
        return "Bad"
    elif 9 <= mean_value < 31:
        return "Average"
    elif 32 <= mean_value < 37:
        return "Good"
    elif 37 <= mean_value < 50:
        return "Excellent"
    else:
        return "Out of range"

# Define cluster storage
to_plot_very_bad = []
to_plot_bad = []
to_plot_average = []
to_plot_good = []
to_plot_excellent = []

# Classify links based on their time series mean
for key, values in data_expe.items():
    cluster = classify_series(values)
    if cluster == "Very Bad":
        to_plot_very_bad.append(key)
    elif cluster == "Bad":
        to_plot_bad.append(key)
    elif cluster == "Average":
        to_plot_average.append(key)
    elif cluster == "Good":
        to_plot_good.append(key)
    elif cluster == "Excellent":
        to_plot_excellent.append(key)

# Collect MAE values per cluster
cluster_dict = {
    "Very Bad": to_plot_very_bad,
    "Bad": to_plot_bad,
    "Average": to_plot_average,
    "Good": to_plot_good,
    "Excellent": to_plot_excellent
}

print(cluster_dict, [len(cluster_dict[category]) for category in cluster_dict])

# Initialize MAE data per cluster
fixed_mae_per_cluster = {cluster: {step: [] for step in steps} for cluster in cluster_dict}
adaptive_mae_per_cluster = {cluster: {step: [] for step in steps} for cluster in cluster_dict}

# Collect MAE values from fixed data
for link in fixed_data:
    for cluster, links in cluster_dict.items():
        if link in links:
            for step in fixed_data[link]:
                mae_value = fixed_data[link][step]['mae']
                if mae_value != 0:
                    fixed_mae_per_cluster[cluster][step].append(mae_value)

# Collect MAE values from adaptive data
for link in adaptive_data:
    for cluster, links in cluster_dict.items():
        if link in links:
            for step in adaptive_data[link]:
                mae_value = adaptive_data[link][step]['mae']
                if mae_value != 0:
                    adaptive_mae_per_cluster[cluster][step].append(mae_value)

# Plotting
#fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid for 4 clusters
#clusters = ["Bad", "Average", "Good", "Excellent"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)  # 2x2 grid for 4 clusters
clusters = ["Average", "Good", "Excellent"]
colors = ["yellow", "red", "orange", "green", "blue"]

for ax, cluster, color in zip(axes.flat, clusters, colors):
    plot_data = []
    
    # Convert to DataFrame for plotting
    for step in steps:
        for value in fixed_mae_per_cluster[cluster][step]:
            plot_data.append({"Step": step, "MAE": value, "Approach": "Fixed"})
        for value in adaptive_mae_per_cluster[cluster][step]:
            plot_data.append({"Step": step, "MAE": value, "Approach": "Adaptive"})
    
    df = pd.DataFrame(plot_data)
    
    if not df.empty:
        df["Step"] = df["Step"].astype(int)
        # Keep only two steps at each time
        df = df[df["Step"].isin([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])]
        sns.boxplot(x="Step", y="MAE", hue="Approach", data=df, ax=ax, palette=["green", "red"], order=sorted(df["Step"].unique()))
        
        ax.set_title(f"{cluster} Links (n={len(cluster_dict[cluster])})")
        ax.set_xlabel("Step")
        ax.set_ylabel("MAE")
        ax.legend(title="Approach")
        ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
#plt.show()
plt.savefig(f"../figures/mae-clusters.pdf", format="pdf", dpi=300)
