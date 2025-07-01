import math
from collections import defaultdict
import json
import time

SLA = 0.85
res = defaultdict(lambda: defaultdict(list))  # res[i][link] = list of txop differences

# Load the series_list from the file
with open('../data/series-iotj-24h.json', 'r') as file:
    observations = json.load(file)

# Load the predictions from the file
with open('../data/adaptive-model-continuous-24h-predictions-merged.json', 'r') as file:
    predictions = json.load(file)


for link, steps in predictions.items():
    x_obs_ = observations.get(link, [])# Get the last len(x_pred) observations for this link
    print(x_obs_)
    for i in range(0, len(x_obs_)):
        for step in range(1, 21):  # Steps from 1 to 10
            print("len: ", len(steps[str("1")]), len(x_obs_))
            x_obs = x_obs_[-len(steps[str("1")]):]  # Ensure x_obs is at least as long as the current step
            print("i: ", i, " step: ", step)
            if i + step > len(steps[str(step)]):
                continue
            """
            for j in range(1, step + 1):
                if steps[str(j)][i] < 0 or steps[str(j)][i] > 50:
                    print(f"Invalid prediction for step {j} at index {i}: {steps[str(j)][i]}")
                    print("new value: ", max(0, min(50, steps[str(j)][i])))
                    time.sleep(1)
            """
            pdr_p = sum(max(0, min(50, steps[str(j)][i])) for j in range(1, step + 1)) / step
            pdr_r = sum(x_obs[i+j] for j in range(0,step)) / step
            
            if link == 'm3-133_m3-153':
                
                for j in range(1,step+1):
                    print("for step: ", j)
                    if i + j >= len(steps[str(j)]):
                        continue
                    print("pred: ", steps[str(j)][i])
                
                for j in range(0,step):
                    print("obs: ", x_obs[i+j])
                #time.sleep(0.01)

            print(step, link, pdr_p, pdr_r)

            pdr_p = pdr_p/50
            pdr_r = pdr_r/50

            txop_r = 0
            txop_p = 0
            if pdr_p == 0:
                continue
            if pdr_r == 0:
                continue
            if pdr_r == 1:
                txop_r = 1
            if pdr_p == 1:
                txop_p = 1

            if pdr_r < 1 and pdr_p < 1:
                print(pdr_r, pdr_p)
                #time.sleep(1)
                txop_r = math.log(1 - SLA) / math.log(1 - pdr_r)
                txop_p = math.log(1 - SLA) / math.log(1 - pdr_p)

            res[step][link].append(txop_r - txop_p)

            print(f"Step: {step}, Link: {link}, i: {i}, PDR_P: {pdr_p:.6f}, PDR_R: {pdr_r:.6f}, "
                  f"TXOP_R: {txop_r:.6f}, TXOP_P: {txop_p:.6f}, "
                  f"Difference: {txop_r - txop_p:.6f}")
            
            if txop_r - txop_p > 100:
                if link == 'm3-133_m3-153':
                    print(f"****Step: {step}, Link: {link}, i: {i}, PDR_P: {pdr_p:.6f}, PDR_R: {pdr_r:.6f}, "
                        f"TXOP_R: {txop_r:.6f}, TXOP_P: {txop_p:.6f}, "
                        f"Difference: {txop_r - txop_p:.6f} < -100")
                    time.sleep(0.01)

print(res)

# Save the results to a file
with open('../data/txop-differences.json', 'w') as file:
    json.dump(res, file, indent=4)