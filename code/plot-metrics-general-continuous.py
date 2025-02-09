import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller

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

# Organize data
fixed_mae = {step: [] for step in steps}
adaptive_mae = {step: [] for step in steps}

cpt = 0
to_plot = ["m3-99_m3-123", "m3-123_m3-99", "m3-123_m3-133", "m3-123_m3-153", "m3-123_m3-159", "m3-123_m3-163", "m3-133_m3-123",
           "m3-133_m3-143", "m3-133_m3-153", "m3-133_m3-159", "m3-133_m3-163",
           "m3-143_m3-123", "m3-143_m3-133", "m3-143_m3-150", "m3-143_m3-153", "m3-143_m3-159",
           "m3-150_m3-143", "m3-150_m3-153", "m3-150_m3-159", "m3-150_m3-163",
           "m3-159_m3-133", "m3-159_m3-143", "m3-159_m3-153", "m3-159_m3-163", "m3-159_m3-166",
           "m3-163_m3-153", "m3-163_m3-159", "m3-163_m3-166", "m3-166_m3-159"]

to_plot = ["m3-143_m3-153", "m3-143_m3-159", "m3-150_m3-143", "m3-150_m3-153", "m3-150_m3-163", "m3-163_m3-153", "m3-166_m3-159"]

# Collect mae values
for link in fixed_data:
    #if link not in to_plot:
        #continue
    for step in fixed_data[link]:
        mae_value = fixed_data[link][step]['mae']
        if mae_value != 0:
            fixed_mae[step].append(mae_value)
            cpt += 1

for link in adaptive_data:
    #if link not in to_plot:
        #continue
    for step in adaptive_data[link]:
        mae_value = adaptive_data[link][step]['mae']
        if mae_value != 0:
            adaptive_mae[step].append(mae_value)

# Convert data to plotting format
plot_data = []
for step in steps:
    for value in fixed_mae[step]:
        plot_data.append({"Step": step, "MAE": value, "Approach": "Fixed"})
    for value in adaptive_mae[step]:
        plot_data.append({"Step": step, "MAE": value, "Approach": "Adaptive"})

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(plot_data)

# Plot
plt.figure(figsize=(10, 6))
df["Step"] = df["Step"].astype(int)  # Convert Step to numeric
sns.boxplot(x="Step", y="MAE", hue="Approach", data=df, palette=["blue", "orange"], order=sorted(df["Step"].unique()))
#plt.scatter(df["Step"], df["MAE"], c=df["Approach"].map({"Fixed": "blue", "Adaptive": "orange"}), alpha=0.5)
#plt.ylim(0, 10)
plt.xlabel("Step")
plt.ylabel("MAE")
plt.title("Comparison of MAE across Steps for Fixed and Adaptive Approaches")
plt.legend(title="Approach")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
