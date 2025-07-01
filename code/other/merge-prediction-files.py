import json

# Output dictionary to hold merged data
merged_data = {}

# Loop over the six files
for i in range(1, 7):
    filename = f"../data/adaptive-model-continuous-24h-predictions-{i}.json"
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                merged_data.update(data)
            else:
                print(f"Warning: {filename} is not a dictionary.")
    except FileNotFoundError:
        print(f"Warning: {filename} not found.")
    except json.JSONDecodeError:
        print(f"Error: {filename} is not valid JSON.")

# Save the merged dictionary to a new file
with open("../data/adaptive-model-continuous-24h-predictions-merged.json", "w") as out_file:
    json.dump(merged_data, out_file, indent=2)

print("Merging completed. Output written to ../data/adaptive-model-continuous-24h-predictions-merged.json.")
