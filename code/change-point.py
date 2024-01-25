import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.use('TkAgg')

# Generate example binomial temporal data (replace this with your actual data)
np.random.seed(42)
n_samples = 200
p_change = 0.1  # Probability of change point
data = np.random.binomial(1, p_change, n_samples)

# Display the original data
plt.figure(figsize=(12, 4))
plt.plot(data, marker='o', linestyle='-', color='b')
plt.title('Original Binomial Temporal Data')
plt.xlabel('Time')
plt.ylabel('Success/Failure')
plt.show()

# Perform change-point analysis using the Pelt method (other methods are available)
model = "binseg"
algo = rpt.Binseg(model=model).fit(data)
result = algo.predict(pen=10)  # Penalties help control the number of change points

# Display the change-point analysis result
rpt.display(data, result, figsize=(12, 4))
plt.title('Change-Point Analysis Result')
plt.show()