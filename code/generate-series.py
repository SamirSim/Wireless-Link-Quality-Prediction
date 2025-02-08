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

# Log data
log_entries = [
    "1738342691.743850;m3-143;SendingBroadcast;0000033",
    "1738342691.754859;m3-153;Data received from fe80::9779 '0000033'",
    "1738342691.757584;m3-150;Data received from fe80::9779 '0000033'",
    "1738342691.767756;m3-159;Data received from fe80::9779 '0000033'",
    "1738342691.778918;m3-166;SendingBroadcast;0000033",
    "1738342691.779225;m3-133;Data received from fe80::9779 '0000033'",
    "1738342691.795417;m3-133;SendingBroadcast;0000033",
    "1738342691.797799;m3-123;SendingBroadcast;0000033",
    "1738342691.799553;m3-159;SendingBroadcast;0000033",
    "1738342691.802446;m3-153;SendingBroadcast;0000033",
    "1738342691.805463;m3-150;SendingBroadcast;0000033",
    "1738342691.812887;m3-163;SendingBroadcast;0000033",
    "1738342691.813587;m3-123;Data received from fe80::2360 '0000033'",
    "1738342691.818374;m3-153;Data received from fe80::b676 '0000033'",
    "1738342691.823685;m3-143;Data received from fe80::b676 '0000033'",
    "1738342691.831638;m3-159;Data received from fe80::9276 '0000033'",
    "1738342691.842814;m3-166;Data received from fe80::9276 '0000033'",
    "1738342691.876909;m3-163;Data received from fe80::a081 '0000033'",
    "1738342691.887843;m3-143;Data received from fe80::a081 '0000033'",
    "1738342691.898419;m3-153;Data received from fe80::a081 '0000033'",
    "1738342691.917704;m3-150;Data received from fe80::b081 '0000033'",
    "1738342691.919916;m3-143;Data received from fe80::b081 '0000033'",
    "1738342691.923446;m3-133;Data received from fe80::b081 '0000033'",
    "1738342691.924896;m3-163;Data received from fe80::b081 '0000033'",
    "1738342691.925698;m3-123;Data received from fe80::b081 '0000033'",
    "1738342692.031690;m3-99;SendingBroadcast;0000033",
    "1738342692.053680;m3-123;Data received from fe80::b277 '0000033'",
    "1738342692.122563;m3-153;Data received from fe80::c276 '0000033'",
    "1738342692.127474;m3-99;Data received from fe80::c276 '0000033'",
    "1738342692.751647;m3-143;SendingBroadcast;0000034",
    "1738342692.755281;m3-133;Data received from fe80::9779 '0000034'",
    "1738342692.762401;m3-153;Data received from fe80::9779 '0000034'",
    "1738342692.765355;m3-150;Data received from fe80::9779 '0000034'",
    "1738342692.770664;m3-166;SendingBroadcast;0000034",
    "1738342692.788689;m3-163;Data received from fe80::9671 '0000034'",
    "1738342692.791633;m3-159;Data received from fe80::9671 '0000034'",
    "1738342692.794211;m3-153;SendingBroadcast;0000034",
    "1738342692.797264;m3-150;SendingBroadcast;0000034",
    "1738342692.803084;m3-133;SendingBroadcast;0000034",
    "1738342692.805695;m3-123;SendingBroadcast;0000034",
    "1738342692.807583;m3-159;SendingBroadcast;0000034",
    "1738342692.815471;m3-143;Data received from fe80::b081 '0000034'",
    "1738342692.820700;m3-163;SendingBroadcast;0000034",
    "1738342692.829317;m3-150;Data received from fe80::b081 '0000034'",
    "1738342692.836666;m3-163;Data received from fe80::b081 '0000034'",
    "1738342692.850669;m3-166;Data received from fe80::9276 '0000034'",
    "1738342692.851049;m3-133;Data received from fe80::b081 '0000034'",
    "1738342692.858327;m3-153;Data received from fe80::9276 '0000034'",
    "1738342692.879547;m3-143;Data received from fe80::b676 '0000034'",
    "1738342692.887679;m3-159;Data received from fe80::9276 '0000034'",
    "1738342692.935752;m3-159;Data received from fe80::c276 '0000034'",
    "1738342692.938432;m3-153;Data received from fe80::c276 '0000034'",
    "1738342692.943483;m3-99;Data received from fe80::c276 '0000034'",
    "1738342692.963345;m3-133;Data received from fe80::c276 '0000034'",
    "1738342692.980802;m3-163;Data received from fe80::a081 '0000034'",
    "1738342692.991660;m3-143;Data received from fe80::a081 '0000034'",
    "1738342692.994748;m3-166;Data received from fe80::a081 '0000034'",
    "1738342693.002284;m3-153;Data received from fe80::a081 '0000034'",
    "1738355311.809007;m3-143;SendingBroadcast;0012653",
    "1738355311.817691;m3-166;SendingBroadcast;0012653",
    "1738355311.820267;m3-159;Data received from fe80::9779 '0012653'",
    "1738355311.821741;m3-153;Data received from fe80::9779 '0012653'",
    "1738355311.824150;m3-133;Data received from fe80::9779 '0012653'",
    "1738355311.835024;m3-150;Data received from fe80::9779 '0012653'",
    "1738355311.849470;m3-123;SendingBroadcast;0012653",
    "1738355311.849716;m3-99;Data received from fe80::c276 '0012653'",
    "1738355311.851323;m3-163;SendingBroadcast;0012653",
    "1738355311.853804;m3-153;SendingBroadcast;0012653",
    "1738355311.856015;m3-133;SendingBroadcast;0012653",
    "1738355311.883163;m3-163;Data received from fe80::a081 '0012653'",
    "1738355311.883976;m3-159;SendingBroadcast;0012653",
    "1738355311.885569;m3-153;Data received from fe80::a081 '0012653'",
    "1738355311.889030;m3-143;Data received from fe80::a081 '0012653'",
    "1738355311.897717;m3-166;Data received from fe80::a081 '0012653'",
    "1738355311.914961;m3-150;SendingBroadcast;0012653",
    "1738355311.937091;m3-143;Data received from fe80::b676 '0012653'",
    "1738355311.952800;m3-143;Data received from fe80::2360 '0012653'",
    "1738355311.961242;m3-123;Data received from fe80::2360 '0012653'",
    "1738355311.980182;m3-159;Data received from fe80::b081 '0012653'",
    "1738355311.994924;m3-150;Data received from fe80::b081 '0012653'",
    "1738355312.011227;m3-163;Data received from fe80::b081 '0012653'",
    "1738355312.025400;m3-123;Data received from fe80::b081 '0012653'",
    "1738355312.105401;m3-99;SendingBroadcast;0012653"
    ]

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