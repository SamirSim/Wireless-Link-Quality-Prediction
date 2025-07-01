import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time
import numpy as np

# Load JSON files (replace with actual file paths)
with open("../data/series-iotj-24h.json", "r") as f:
    data_expe = json.load(f)

def classify_series(time_series):
    mean_value = np.mean(time_series)

    if 0 < mean_value < 9:
        return "Bad"
    elif 9 <= mean_value < 32:
        return "Average"
    elif 32 <= mean_value < 37:
        return "Good"
    elif 37 <= mean_value <= 50:
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

# Load the results from the file
with open('../data/txop-differences.json', 'r') as file:
    res = json.load(file)

# Flatten the res dict into a long DataFrame with columns: step, value, link
data = []

for step, links in res.items():
    if int(step) > 20:
        continue
    for link, values in links.items():
        for value in values:
            data.append({"step": int(step), "value": value, "link": link})

df = pd.DataFrame(data)

# Map each link to its cluster
link_to_cluster = {}
for cluster_name, link_list in cluster_dict.items():
    for link in link_list:
        link_to_cluster[link] = cluster_name

df["cluster"] = df["link"].map(link_to_cluster)

# Drop rows with no cluster (if any)
df = df.dropna(subset=["cluster"])

import matplotlib.pyplot as plt
import seaborn as sns

clusters = df["cluster"].unique()

counts = df.groupby(["cluster", "step"]).size().reset_index(name="count")

# Print in a clean format
for cluster in sorted(counts["cluster"].unique()):
    print(f"\nCluster: {cluster}")
    subset = counts[counts["cluster"] == cluster]
    for _, row in subset.iterrows():
        print(f"  Step {int(row['step'])}: {int(row['count'])} values")

# One plot per cluster
for cluster in sorted(clusters):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[df["cluster"] == cluster],
                x="step", y="value", palette="muted", showfliers=True)
    plt.xlabel("Step")
    plt.ylabel("TxOp Difference (txop_r - txop_p)")
    plt.title(f"TxOp Differences Across Steps — Cluster: {cluster} (n={len(cluster_dict[cluster])})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from collections import defaultdict

average_df = df[df["cluster"] == "Average"]

# Optional: sort by step
average_df = average_df.sort_values(by="step")
print(average_df)
fliers_proportion = {}

# Group by step (or your category variable)
for step, group in df.groupby("step"):
    values = group["value"]
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    is_flier =  (values > upper_bound)
    prop = is_flier.sum() / len(values)
    fliers_proportion[step] = prop

# Optional: print nicely
print(f"{'Step':>5} | {'% Fliers':>9}")
print("-" * 18)
for step in sorted(fliers_proportion):
    print(f"{step:5} | {fliers_proportion[step]*100:8.2f}%")