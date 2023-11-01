import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# Data preparation
# Original data
yellow_points = np.array([[1,2], [2,1], [2,3], [4,3]])
black_points = np.array([[6,5], [7,8], [8,3], [8,4]])
all_points = np.vstack((yellow_points, black_points))
labels = np.array([1, 1, 1, 1, -1, -1, -1, -1, -1])  # Update labels with the new sample

# Train SVM with C=1
clf_C1 = svm.SVC(kernel='linear', C=1)
clf_C1.fit(all_points, labels)

# Train SVM with C=∞ (approximated with a large number)
clf_Cinf = svm.SVC(kernel='linear', C=1e10)
clf_Cinf.fit(all_points, labels)

# Plot setup
plt.figure(figsize=(15, 7))
ax = plt.subplot(1, 2, 1)
plt.scatter(yellow_points[:, 0], yellow_points[:, 1], color='yellow', s=30, label='Yellow Points')
plt.scatter(black_points[:, 0], black_points[:, 1], color='black', s=30, label='Black Points')

# Plot decision boundary and margins for C=1
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 50)
yy = np.linspace(ylim[0], ylim[1], 50)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z_C1 = clf_C1.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z_C1, colors='blue', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title('C=1')

# Plot decision boundary and margins for C=∞
ax = plt.subplot(1, 2, 2)
plt.scatter(yellow_points[:, 0], yellow_points[:, 1], color='yellow', s=30, label='Yellow Points')
plt.scatter(black_points[:, 0], black_points[:, 1], color='black', s=30, label='Black Points')
Z_Cinf = clf_Cinf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z_Cinf, colors='blue', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title('C=∞')

# Common plot settings
for ax in plt.gcf().axes:
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.suptitle('Decision Boundary with Different C Values')
plt.show()
