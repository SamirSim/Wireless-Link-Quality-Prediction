import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import sys
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from pandas import read_csv # type: ignore
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore
from statsmodels.tsa.arima.model import ARIMA # type: ignore
from math import sqrt
from pandas import DataFrame # type: ignore
import time
import warnings
import random
import string
import json
from statsmodels.tsa.stattools import adfuller # type: ignore

random.seed(10)

from itertools import permutations

# Load JSON files (replace with actual file paths)
with open("../data/series-iotj-24h.json", "r") as f:
    data_expe = json.load(f)

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

# Generate all possible sender-receiver pairs without "m3"
couples = [(int(sender[3:]), int(receiver[3:]))  for sender, receiver in permutations(node_mac_map.keys(), 2)]

# Assuming data_expe, data_simu, data_simu_pdr, and couples are defined elsewhere in your code
# couples is a list of tuples where each tuple contains (n, m)

# Adjust the following constants as needed
subplots_per_figure = 25
rows = 5  # Adjust based on the desired subplot arrangement
cols = 5  # Adjust based on the desired subplot arrangement

num_figures = (len(couples) + subplots_per_figure - 1) // subplots_per_figure

#plt.rcParams.update({'font.size': 17})
for fig_num in range(num_figures):
    fig, axes = plt.subplots(rows, cols, figsize=(27, 8))
    
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing

    for subplot_index in range(subplots_per_figure):
        index = fig_num * subplots_per_figure + subplot_index
        if index >= len(couples):
            break

        n, m = couples[index]
        sender = "m3-" + str(n)
        receiver = "m3-" + str(m)
        key = sender + "_" + receiver
        data_expe_values = data_expe[key]

        y_exp = np.array(data_expe_values)

        ax = axes[subplot_index]
        
        ax.set_title(key)
        ax.plot(y_exp, label='experiments')
        ax.set_xlabel('Time Window')

        if fig_num == 0 and subplot_index == 0:
            ax.set_ylabel('Number of packets received')
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels)

    plt.show()
    #plt.savefig("Links.pdf", format="pdf", bbox_inches="tight")
    #sys.exit(0)