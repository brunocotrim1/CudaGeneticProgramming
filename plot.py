import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
rows = np.array([1000000, 1500000, 1800000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 10000000])
gpu = np.array([38.895, 42.89, 42.686, 41.112, 42.981, 43.627, 42.137, 47.160, 47.629, 49.571])
cpu = np.array([19.872, 36.095, 42.095, 46.770, 68.424, 101.630, 124.090, 164.056, 188.512, 277.953])

# Reshape data for sklearn
rows_reshape = rows.reshape(-1, 1)
gpu_reshape = gpu.reshape(-1, 1)
cpu_reshape = cpu.reshape(-1, 1)

# Create linear regression models
gpu_model = LinearRegression().fit(rows_reshape, gpu_reshape)
cpu_model = LinearRegression().fit(rows_reshape, cpu_reshape)

# Predictions
gpu_pred = gpu_model.predict(rows_reshape)
cpu_pred = cpu_model.predict(rows_reshape)

# Plotting
plt.scatter(rows, gpu, label='GPU')
plt.scatter(rows, cpu, label='CPU')
plt.plot(rows, gpu_pred, label='GPU Regression Line', linestyle='--')
plt.plot(rows, cpu_pred, label='CPU Regression Line', linestyle='--')

plt.xlabel('Rows')
plt.ylabel('Time')
plt.title('GPU and CPU Performance')
plt.legend()
plt.show()
