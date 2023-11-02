import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# Function to plot the decision function for a 2D SVC
def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)
    
    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    # Plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
                   s=200, linewidth=1, facecolors='none', edgecolors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Function to train SVM
def train_svm(data, labels, C_value):
    clf = svm.SVC(kernel='linear', C=C_value)
    clf.fit(data, labels)
    return clf

# Function to plot decision boundary and support vectors
def plot_decision_boundary(clf, points, labels, title, additional_points=None, test_samples=None):
    plt.figure(figsize=(10, 7))
    
    # Plot data points
    plt.scatter(points[:, 0], points[:, 1], s=30, c=labels, cmap=plt.cm.Paired, label='Training Points')
    
    # Plot additional points
    if additional_points is not None:
        for point, label in additional_points:
            plt.scatter(point[0], point[1], s=50, color='red' if label == 1 else 'blue', 
                        label=f'Added Point (Class {label})')
            points = np.vstack((points, point))
            labels = np.append(labels, label)
    
    # Plot test samples
    if test_samples is not None:
        for sample in test_samples:
            test_point, color = sample
            test_label = clf.predict([test_point])[0]
            plt.scatter(test_point[0], test_point[1], s=50, color=color, 
                        label=f'Test Point {test_point} (Class {test_label})')

    plot_svc_decision_function(clf)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(title)
    plt.legend()

# Main function
def main():
    # Initial data setup
    points = np.array([[1,2], [2,1], [2,3], [4,3], [6,5], [7,8], [8,3], [8,4]])
    labels = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    
    # Scenarios
    scenarios = [
        {'C': 1e6, 'additional_points': [], 'test_samples': [], 'title': 'Initial Decision Boundary'},
        {'C': 1e6, 'additional_points': [(np.array([7,5]), -1)], 'test_samples': [], 'title': 'Decision Boundary with Black (7,5)'},
        {'C': 1e6, 'additional_points': [(np.array([4,2]), 1)], 'test_samples': [], 'title': 'Decision Boundary with Yellow (4,2)'},
        {'C': 1e6, 'additional_points': [], 'test_samples': [([7,5], 'black'), ([6,4], 'black'), ([4,2], 'yellow'), ([5,3], 'yellow'), ([5,3], 'black'), ([6,4], 'yellow')], 'title': 'Decision Boundary with Test Samples'},
        {'C': 1, 'additional_points': [(np.array([4,4]), -1)], 'test_samples': [], 'title': 'Decision Boundary with Black (4,4) & C=1'},
        {'C': 1e6, 'additional_points': [(np.array([4,4]), -1)], 'test_samples': [], 'title': 'Decision Boundary with Black (4,4) & C=1e6'}
    ]
    
    # Run scenarios
    for scenario in scenarios:
        clf = train_svm(points, labels, C_value=scenario['C'])
        plot_decision_boundary(clf, points, labels, scenario['title'], scenario['additional_points'], scenario['test_samples'])
    
    plt.show()

main()
