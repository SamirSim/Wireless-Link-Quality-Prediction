from collections import defaultdict
import bisect
import json

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
with open("../data/logs-iotj-24h.txt", "r") as file:
    log_entries = file.readlines()
    log_entries = [line.strip() for line in log_entries]

# Dictionary to track received packets per sender
packet_reception = defaultdict(list)  # {sender: [(receiver, packet_id, timestamp), ...]}

# Dictionary to track packet ID sent by each sender
sent_packets = []  # [(sender, packet_id, timestamp)]

for entry in log_entries:
    parts = entry.split(";")
    timestamp = float(parts[0])
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

print(windowed_reception)

# Save results to file
with open("../data/series-iotj-24h.json", "w") as file:
    json.dump(windowed_reception, file)