import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('build/l2_errors.txt')
data = np.log(data)

slope, intercept = np.polyfit(data[:, 0], data[:, 1], 1)
print("slope =", slope)

# plot
plt.figure(figsize=(7, 5))
plt.scatter(data[:, 0], data[:, 1], label='data points')
plt.plot(data[:, 0], slope * data[:, 0] + intercept, label=f'fit: slope = {slope:.3f}', linewidth=2)

plt.xlabel('log(h)')
plt.ylabel('log(L2 error)')
plt.legend()

plt.show()

