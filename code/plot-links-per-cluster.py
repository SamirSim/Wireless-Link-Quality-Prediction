import json
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import seaborn as sns # type: ignore

sns.set_style("whitegrid")
sns.color_palette("tab10")

# Font size
plt.rcParams.update({'font.size': 15})

# Load JSON file (replace with actual file path)
with open("../data/series-iotj-24h.json", "r") as f:
    data_expe = json.load(f)



# Extract unique time steps
steps = sorted({step for series in data_expe.values() for step in range(len(series))})

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
        print(mean_value)
        return "Out of range"

# Group links into clusters
clusters = {"Bad": [], "Average": [], "Good": [], "Excellent": []}
print(len(data_expe))
for key, values in data_expe.items():
    cluster = classify_series(values)
    if cluster in clusters:
        clusters[cluster].append(key)

print(clusters)
# Compute statistics (mean, min, max)
stats = {}
for cluster in clusters:
    stats[cluster] = {
        "mean": [np.mean([data_expe[link][step]*2 for link in clusters[cluster] if step < len(data_expe[link])]) for step in steps],
        "min": [np.min([data_expe[link][step]*2 for link in clusters[cluster] if step < len(data_expe[link])]) for step in steps],
        "max": [np.max([data_expe[link][step]*2 for link in clusters[cluster] if step < len(data_expe[link])]) for step in steps],
    }

"""
# Plot results
fig, axes = plt.subplots(1, len(clusters), figsize=(15, 5), sharex=True, sharey=True)

for ax, cluster in zip(axes, clusters):
    ax.plot(steps, stats[cluster]["mean"], label="Mean", color="blue")
    ax.fill_between(steps, stats[cluster]["min"], stats[cluster]["max"], color="blue", alpha=0.2)
    ax.set_title(f"{cluster} Links (n={len(clusters[cluster])})")
    #ax.set_xlabel("Time interval (x T=50 seconds)")
    ax.set_ylabel("Value")
    ax.legend()
fig.text(0.5, 0.01, "Time interval (x T=50 seconds)", ha="center")
plt.suptitle("Series Evolution per Cluster")
plt.tight_layout()
plt.savefig(f"../figures/links-clusters.pdf", format="pdf", dpi=300)
#plt.show()
"""
# Plot and save figures
for cluster in clusters:
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(steps, stats[cluster]["mean"], label="Mean", color="blue")
    ax.fill_between(steps, stats[cluster]["min"], stats[cluster]["max"], color="blue", alpha=0.2)

    #ax.set_title(f"{cluster} Links (n={len(clusters[cluster])})")
    # Uncomment the following line if you want to add X-axis label
    ax.set_xlabel("Time interval (x T=50 seconds)")

    ax.set_ylabel("PDR (%)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"../figures/{cluster.lower()}_cluster_plot.pdf", format="pdf")
    plt.close(fig)  # Close the figure to avoid display if running interactively