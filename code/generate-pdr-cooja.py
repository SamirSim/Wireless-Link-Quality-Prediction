import json
import random
from itertools import permutations
import numpy as np # type: ignore

WINDOW_SIZE = 50
random.seed(10)

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
couples = [(int(sender[3:]), int(receiver[3:])) for sender, receiver in permutations(node_mac_map.keys(), 2)]

# Function to calculate PDR for each sender-receiver pair
def calculate_pdr(couples, data_expe, window_size=WINDOW_SIZE):
    success_ratios = {}
    
    for sender_idx, receiver_idx in couples:
        sender = f"m3-{sender_idx}"
        receiver = f"m3-{receiver_idx}"
        pair_key = f"{sender}_{receiver}"
        
        # Check if the pair exists in the data_expe dictionary
        if pair_key in data_expe:
            # Retrieve the received packets for this pair (if the key exists)
            received_packets = data_expe[pair_key]
            
            # Calculate the PDR based on the window size
            received_count = np.mean(received_packets)
            pdr = received_count / window_size
            
            success_ratios[f'"{sender}_{receiver}"'] = pdr
    
    return success_ratios

# Calculate PDR for all couples
success_ratios = calculate_pdr(couples, data_expe)

# Generate the output in the desired format
output = []
for pair, pdr in success_ratios.items():
    output.append(f'successRatios.put({pair}, {pdr});')

# Print the formatted output
print("\n".join(output))
