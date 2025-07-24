import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity

# Example non-normally distributed data
original_data = np.random.exponential(scale=2, size=1000)

# Fit kernel density estimation to the original data
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(original_data.reshape(-1, 1))

# Generate synthetic data using kernel density estimation
synthetic_data = kde.sample(1000).flatten()

# Plot the original and synthetic data
sns.histplot(original_data, kde=True, label='Original Data', color='blue', alpha=0.7)
sns.histplot(synthetic_data, kde=True, label='Synthetic Data', color='orange', alpha=0.7)
plt.title('Kernel Density Estimation for Synthetic Data Generation')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()
