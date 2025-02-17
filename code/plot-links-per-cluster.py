import json
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import seaborn as sns # type: ignore

sns.set_style("whitegrid")
sns.color_palette("tab10")

# Load JSON file (replace with actual file path)
with open("../data/series-iotj-24h.json", "r") as f:
    data_expe = json.load(f)

# Extract unique time steps
steps = sorted({step for series in data_expe.values() for step in range(len(series))})

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

# Group links into clusters
clusters = {"Bad": [], "Average": [], "Good": [], "Excellent": []}
for key, values in data_expe.items():
    cluster = classify_series(values)
    if cluster in clusters:
        clusters[cluster].append(key)

# Compute statistics (mean, min, max)
stats = {}
for cluster in clusters:
    stats[cluster] = {
        "mean": [np.mean([data_expe[link][step] for link in clusters[cluster] if step < len(data_expe[link])]) for step in steps],
        "min": [np.min([data_expe[link][step] for link in clusters[cluster] if step < len(data_expe[link])]) for step in steps],
        "max": [np.max([data_expe[link][step] for link in clusters[cluster] if step < len(data_expe[link])]) for step in steps],
    }

# Plot results
fig, axes = plt.subplots(1, len(clusters), figsize=(15, 5), sharex=True, sharey=True)

for ax, cluster in zip(axes, clusters):
    ax.plot(steps, stats[cluster]["mean"], label="Mean", color="blue")
    ax.fill_between(steps, stats[cluster]["min"], stats[cluster]["max"], color="blue", alpha=0.2)
    ax.set_title(f"{cluster} Links (n={len(clusters[cluster])})")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Value")
    ax.legend()

plt.suptitle("Series Evolution per Cluster")
plt.tight_layout()
plt.savefig(f"../figures/links-clusters.pdf", format="pdf", dpi=300)
#plt.show()