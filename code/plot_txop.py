import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time

# Load the results from the file
with open('../data/txop-differences.json', 'r') as file:
    res = json.load(file)

# Flatten the res dict into a long DataFrame with columns: step, value
data = []

for step, links in res.items():
    if int(step) > 20:  # Only consider steps 1 to 10
        continue
    for link, values in links.items():
        for value in values:
            data.append({"step": step, "value": value})

df = pd.DataFrame(data)

# Optional: sort steps numerically
df["step"] = pd.to_numeric(df["step"])

for step in df["step"].unique():
    print(f"Step {step}: {len(df[df['step'] == step])} txop differences")

print(df)
# Plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x="step", y="value", inner="quartile", palette="muted")
sns.boxplot(data=df, x="step", y="value", palette="muted", showfliers=True)
plt.xlabel("Step")
plt.ylabel("TxOp Difference (txop_r - txop_p)")
plt.title("Distribution of TxOp Differences Across Steps")
plt.grid(True)
plt.yscale('log')
plt.tight_layout()
plt.show()

from collections import defaultdict

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

time.sleep(5)
# Initialize counters
summary = defaultdict(lambda: {"< 0": 0, "0 to 5": 0, "> 5": 0, "Total": 0})

for step, links in res.items():
    for link, values in links.items():
        for value in values:
            summary[step]["Total"] += 1
            if value < 0:
                summary[step]["< 0"] += 1
                print(f"Step {step}, Link {link}, Value {value} < 0")
            elif value > 2:
                summary[step]["> 5"] += 1
            else:
                summary[step]["0 to 5"] += 1

# Sort steps numerically
sorted_steps = sorted(summary.keys(), key=int)

# Print table header
print(f"{'Step':>4} | {'< 0':>7} | {'0 to 5':>2} | {'> 5':>7}")
print("-" * 40)

# Print row for each sorted step with proportions
for step in sorted_steps:
    total = summary[step]["Total"]
    a = summary[step]["< 0"] / total if total else 0
    b = summary[step]["0 to 5"] / total if total else 0
    c = summary[step]["> 5"] / total if total else 0
    print(f"{step:>4} | {a:7.2%} | {b:2.2%} | {c:7.2%}")
