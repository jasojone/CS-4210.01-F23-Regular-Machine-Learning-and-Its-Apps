import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# Data preparation
yellow_points = np.array([[1,2], [2,1], [2,3], [4,2], [4,3]])
black_points = np.array([[6,5], [7,5], [7,8], [8,3], [8,4]])
all_points = np.vstack((yellow_points, black_points))
labels = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

# SVM training
clf = svm.SVC(kernel='linear', C=1e6)
clf.fit(all_points, labels)

# Plot setup
plt.figure(figsize=(10, 7))
ax = plt.gca()
plt.scatter(yellow_points[:, 0], yellow_points[:, 1], color='yellow', s=30, label='Yellow Points')
plt.scatter(black_points[:, 0], black_points[:, 1], color='black', s=30, label='Black Points')

# Plot decision boundary and margins
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 50)
yy = np.linspace(ylim[0], ylim[1], 50)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='blue', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Test samples
# test_positions = [(7,5), (6,4), (4,2), (5,3), (5,3), (6,4)]
# test_colors = ['black', 'black', 'yellow', 'yellow', 'black', 'yellow']
# test_positions = [(7,5), (6,4), (5,3)]
# test_colors = ['black', 'black', 'black']
# test_positions = [(4,2), (5,3), (6,4)]
# test_colors = ['yellow', 'yellow','yellow']

test_labels = clf.predict(test_positions)
for position, true_color, predicted_label in zip(test_positions, test_colors, test_labels):
    marker = 'x' if (true_color == 'yellow' and predicted_label == -1) or (true_color == 'black' and predicted_label == 1) else 'o'
    plt.scatter(position[0], position[1], color=true_color, marker=marker, s=50, edgecolors='gray')
# Add grid lines
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Display
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Decision Boundary, Support Vectors & Test Samples')
plt.show()
