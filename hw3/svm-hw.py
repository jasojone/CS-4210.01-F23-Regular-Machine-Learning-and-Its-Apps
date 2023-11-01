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

# Plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='red', label='Support Vectors')
print("The values of the support vectors are: ")
print(clf.support_vectors_)

# Plotting test samples
test_points = np.array([[4,2], [4,4], [5,3], [6,4], [7,5]])
test_labels = clf.predict(test_points)
yellow_added = False
black_added = False

for point, label in zip(test_points, test_labels):
    if label == 1 and not yellow_added:  # Yellow class
        plt.scatter(point[0], point[1], color='yellow', marker='s', s=50, edgecolors='gray', label='Test Yellow')
        yellow_added = True
    elif label == 1:
        plt.scatter(point[0], point[1], color='yellow', marker='s', s=50, edgecolors='gray')
    elif label == -1 and not black_added:  # Black class
        plt.scatter(point[0], point[1], color='black', marker='s', s=50, edgecolors='gray', label='Test Black')
        black_added = True
    else:
        plt.scatter(point[0], point[1], color='black', marker='s', s=50, edgecolors='gray')

# Display
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Decision Boundary & Support Vectors')
plt.show()
