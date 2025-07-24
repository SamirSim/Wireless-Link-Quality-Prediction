import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import json
import seaborn as sns # type: ignore

sns.set_style("whitegrid")
sns.color_palette("tab10")

# Font size
plt.rcParams.update({'font.size': 15})

# Load JSON file
with open("../data/series-iotj-24h.json", "r") as f:
    data_expe = json.load(f)

# Load JSON files from Cooja logs
with open("../data/series-cooja-iotj-24h.json", "r") as f:
    data_cooja = json.load(f)

# Node MAC mapping
node_mac_map = {
    "m3-99": "b277",
    "m3-123": "c276",
    "m3-133": "2360",
    "m3-143": "9779",
    "m3-150": "b676",
    "m3-153": "b081",
    "m3-159": "a081",
    "m3-163": "9276",
    "m3-166": "9671",
}

couples = [(150, 163), (133, 153), (166, 163)]

# Compute the overall mean and variance across all link data
all_values = []
for key, values in data_expe.items():
    all_values.extend(values)

"""
subplots_per_figure = 3
rows, cols = 1, 3
num_figures = (len(couples) + subplots_per_figure - 1) // subplots_per_figure

for fig_num in range(num_figures):
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    axes = axes.flatten()

    for subplot_index in range(subplots_per_figure):
        index = fig_num * subplots_per_figure + subplot_index
        if index >= len(couples):
            break

        n, m = couples[index]
        sender = "m3-" + str(n)
        receiver = "m3-" + str(m)
        key = sender + "_" + receiver

        if key not in data_expe:
            continue

        y_exp = np.array(data_expe[key])
        y_cooja = np.array(data_cooja[key])
        
        ax = axes[subplot_index]
        ax.set_title(key)
        ax.plot(y_exp, label="Experiments")
        #ax.plot(y_cooja, label="Simulation")
        print(fig_num, subplot_index)
        if fig_num == 0 and subplot_index == 0:
            ax.set_ylabel("# of received packets")
            #ax.set_xlabel("Time interval (T=50 seconds)")
            handles, labels = ax.get_legend_handles_labels()
            #fig.legend(handles, labels, loc="upper right")
            ax.legend()

    # Set a single x-axis label for the entire figure
    fig.text(0.5, 0.01, "Time interval (x T=50 seconds)", ha="center")
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"../figures/links-comparison-audition-expe.pdf", format="pdf", dpi=300)
"""


# Plot each couple individually
for n, m in couples:
    sender = f"m3-{n}"
    receiver = f"m3-{m}"
    key = f"{sender}_{receiver}"

    if key not in data_expe:
        continue

    y_exp = np.array(data_expe[key])
    y_cooja = np.array(data_cooja.get(key, []))  # Use empty array if missing

    fig, ax = plt.subplots(figsize=(6, 3))
    #ax.set_title(f"{key}")
    ax.plot(y_exp*2, label="Experiments")
    ax.plot(y_cooja*2, label="Simulation")  # Uncomment if needed

    ax.set_ylabel("PDR (%)")
    ax.set_xlabel("Time interval (T=50 seconds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"../figures/{key}.pdf", format="pdf")
    plt.close(fig)