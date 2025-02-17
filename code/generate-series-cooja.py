from collections import defaultdict
import bisect
import json
import numpy as np #type: ignore

def convert_time_to_seconds_milliseconds(time_str):
    # Used for translating the timing format from Cooja's to FIT IoT-Lab's 
    # Split the time string into hours, minutes, seconds, and milliseconds
    parts = time_str.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        print(len(parts), parts)
        raise ValueError("Invalid time format")

    # Convert to integers
    hours = int(hours)
    minutes = int(minutes)
    seconds = float(seconds)

    # Convert hours to seconds
    total_seconds = hours * 3600

    # Add minutes to total seconds
    total_seconds += minutes * 60

    # Add remaining seconds
    total_seconds += seconds

    return total_seconds

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

WINDOW_SIZE = 50  # Define time window size

# Read log data from file
with open("../data/cooja-iotj-24h.txt", "r") as file:
    log_entries = file.readlines()
    log_entries = [line.strip() for line in log_entries]

# Dictionary to track received packets per sender
packet_reception = defaultdict(list)  # {sender: [(receiver, packet_id, timestamp), ...]}

# Dictionary to track packet ID sent by each sender
sent_packets = []  # [(sender, packet_id, timestamp)]

for entry in log_entries:
    parts = entry.split(";")
    timestamp = convert_time_to_seconds_milliseconds(parts[0])
    node_id = parts[1]
    event = parts[2]
    
    if "SendingBroadcast" in event:
        packet_id = parts[-1]  # Extract packet ID
        sent_packets.append((node_id, packet_id, timestamp))
    
    elif "Data received from" in event:
        sender_mac = event.split("fe80::")[1].split(" '")[0]
        packet_id = event.split("'")[1]
        
        # Find sender node ID from MAC
        sender_node = None
        for node, mac in node_mac_map.items():
            if mac == sender_mac:
                sender_node = node
                break
        
        if sender_node:
            packet_reception[sender_node].append((node_id, packet_id, timestamp))

# Dictionary to store reception count per time window
windowed_reception = defaultdict(list)  # {"sender_receiver": [count per window]}

time_start = min(ts for _, _, ts in sent_packets) if sent_packets else 0

while sent_packets:
    print(f"Processing window starting at {time_start}")
    time_end = time_start + WINDOW_SIZE
    window_data = [p for p in sent_packets if time_start <= p[2] < time_end]
    sent_packets = [p for p in sent_packets if p[2] >= time_end]
    
    reception_count = defaultdict(int)  # Count successful receptions per sender-receiver
    
    for sender, packet_id, send_time in window_data:
        received_nodes = {recv for recv, pid, _ in packet_reception.get(sender, []) if pid == packet_id}
        
        for node in node_mac_map.keys():
            if node == sender:
                continue
            key = f"{sender}_{node}"
            if node in received_nodes:
                reception_count[key] += 1
    
    # Store results in the windowed dictionary
    for key in node_mac_map.keys():
        for node in node_mac_map.keys():
            if key != node:
                pair_key = f"{key}_{node}"
                windowed_reception[pair_key].append(reception_count.get(pair_key, 0))
    
    time_start = time_end

new_dict = {}
for key, values in windowed_reception.items():
    # Calculate the mean and standard deviation of the original series
    original_series = values
    mean = np.mean(original_series)
    std_dev = np.std(original_series)

    # Define the desired length of the series (1750 elements)
    desired_length = 1750

    # Calculate how many elements we need to generate
    elements_needed = desired_length - len(original_series)

    # Generate additional data points using the same normal distribution
    print(f"Generating {elements_needed} additional data points for {pair_key} for original series: {original_series}")
    additional_data = np.abs(np.random.normal(loc=mean, scale=std_dev, size=elements_needed))
    additional_data = np.round(additional_data)

    # Combine the original series with the new data
    extended_series = original_series + list(additional_data)

    # Store the extended series back into the windowed dictionary
    new_dict[key] = extended_series

print(new_dict)

# Save results to file
with open("../data/series-cooja-iotj-24h.json", "w") as file:
    json.dump(new_dict, file)