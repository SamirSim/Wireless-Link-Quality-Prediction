import json
import pandas as pd
import numpy as np
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
sns.set_style("whitegrid")
# Set color palette
sns.color_palette("tab10")
# Font size
plt.rcParams.update({'font.size': 15})

# Load JSON file of prediction times for the adaptive model
with open("../data/adaptive-times-clusters.json", "r") as f:
    data_adaptive_times = json.load(f)

# Load JSON file of prediction times for the best model
with open("../data/best-times-clusters.json", "r") as f:
    data_best_times = json.load(f)

# Transform the data
summed_data = {
    cluster: [sum(times) for times in couples.values()]
    for cluster, couples in data_adaptive_times.items()
}

# Result
print(summed_data)

# Convert both dictionaries into a DataFrame-friendly format
def prepare_dataframe(data, label):
    rows = []
    for cluster, values in data.items():
        for v in values:
            rows.append({'Cluster': cluster, 'Prediction Time': v, 'Source': label})
    return pd.DataFrame(rows)

df1 = prepare_dataframe(summed_data, 'Adaptive Model')
df2 = prepare_dataframe(data_best_times, 'Best Model')

# Combine into a single DataFrame
df_all = pd.concat([df1, df2])

#df_all = df_all[df_all['Cluster'] != 'Bad']

# Optional: ensure consistent cluster ordering
cluster_order = ['Bad', 'Average', 'Good', 'Excellent']

# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Prediction Time', hue='Source', data=df_all, order=cluster_order)

plt.title("Comparison of Prediction Times by Cluster")
plt.ylabel("Summed Prediction Time")
plt.xlabel("Cluster")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Dataset")
plt.tight_layout()
plt.yscale('log')  # Use logarithmic scale for better visibility of differences
plt.show()

# Print the dataframes means for each cluster
print("Means of prediction times for each cluster:")
for cluster in cluster_order:
    mean_adaptive = df1[df1['Cluster'] == cluster]['Prediction Time'].mean()
    mean_best = df2[df2['Cluster'] == cluster]['Prediction Time'].mean()
    print(f"{cluster}: Adaptive Model = {mean_adaptive:.2f}, Best Model = {mean_best}")
# Print the overall means
print("\nOverall means of prediction times:")
overall_mean_adaptive = df1['Prediction Time'].mean()
overall_mean_best = df2['Prediction Time'].mean()
print(f"Adaptive Model = {overall_mean_adaptive:.2f}, Best Model = {overall_mean_best}")

# Group by Cluster and Source, then describe
summary_table = df_all.groupby(['Cluster', 'Source'])['Prediction Time'].describe()[['min', '25%', '50%', '75%', 'max']]

# Optional: round for display
summary_table = summary_table.round(4)

# Display
print(summary_table)