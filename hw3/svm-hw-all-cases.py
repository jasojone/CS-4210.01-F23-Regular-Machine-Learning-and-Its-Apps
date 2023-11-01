import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC 
    example from https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)
    
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=200, linewidth=1, facecolors='none', edgecolors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Original dataset
yellow_points = np.array([[1,2], [2,1], [2,3], [4,3]])
black_points = np.array([[6,5], [7,8], [8,3], [8,4]])
all_points = np.vstack((yellow_points, black_points))
labels = np.array([1, 1, 1, 1, -1, -1, -1, -1])

# SVM training
clf = svm.SVC(kernel='linear', C=1e6) # C=∞ (approximated with a large number) this is a hard margin
clf.fit(all_points, labels)

# Plot initial decision boundary and support vectors
plt.figure(figsize=(10, 7))
plt.scatter(yellow_points[:, 0], yellow_points[:, 1], color='yellow', s=30, label='Yellow Points')
plt.scatter(black_points[:, 0], black_points[:, 1], color='black', s=30, label='Black Points')
plot_svc_decision_function(clf, plot_support=True)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Initial Decision Boundary & Support Vectors')
plt.legend()
plt.show()

# Function to update plot with new samples and print decision boundary
# Function to update plot with new samples and print decision boundary
def update_plot(new_yellow=None, new_black=None, C_value=1e6, title_suffix=''):
    global yellow_points, black_points, all_points, labels
    if new_yellow is not None:
        yellow_points = np.vstack((yellow_points, new_yellow))
        all_points = np.vstack((all_points, new_yellow))
        labels = np.append(labels, 1)
    if new_black is not None:
        black_points = np.vstack((black_points, new_black))
        all_points = np.vstack((all_points, new_black))
        labels = np.append(labels, -1)
    
    # SVM training
    clf = svm.SVC(kernel='linear', C=C_value)
    clf.fit(all_points, labels)
    
    # Extracting the weight vector (w) and bias (b)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    
    # Decision boundary equation
    decision_boundary_eq = plt.text(0.05, 0.85,"Decision boundary equation: {:.3f}x + {:.3f}y + {:.3f} = 0".format(w[0], w[1], b))
    
    # Printing the decision boundary equation in the terminal
    print(decision_boundary_eq)

    # Plot updated decision boundary and support vectors
    plt.figure(figsize=(10, 7))
    plt.scatter(yellow_points[:, 0], yellow_points[:, 1], color='yellow', s=30, label='Yellow Points')
    plt.scatter(black_points[:, 0], black_points[:, 1], color='black', s=30, label='Black Points')
    plot_svc_decision_function(clf, plot_support=True)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'Updated Decision Boundary & Support Vectors {title_suffix}')
    plt.legend()

    # Add the decision boundary equation as text
    plt.text(0.05, 0.85, "Decision boundary equation: {:.3f}x + {:.3f}y + {:.3f} = 0".format(w[0], w[1], b), transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


# Add new training samples and update plot
update_plot(new_black=[7,5], title_suffix='with Black (7,5)')
update_plot(new_yellow=[4,2], title_suffix='with Yellow (4,2)')

# Test samples
test_samples = {
    'd': {'point': [7,5], 'color': 'black'},
    'e': {'point': [6,4], 'color': 'black'},
    'f': {'point': [4,2], 'color': 'yellow'},
    'g': {'point': [5,3], 'color': 'yellow'},
    'h': {'point': [5,3], 'color': 'black'},
    'i': {'point': [6,4], 'color': 'yellow'}
}

# Plot test samples
plt.figure(figsize=(10, 7))
plt.scatter(yellow_points[:, 0], yellow_points[:, 1], color='yellow', s=30, label='Yellow Points')
plt.scatter(black_points[:, 0], black_points[:, 1], color='black', s=30, label='Black Points')
plot_svc_decision_function(clf, plot_support=True)
for label, sample in test_samples.items():
    plt.scatter(sample['point'][0], sample['point'][1], color=sample['color'], marker='s', s=50, edgecolors='gray', label=f'Test Sample {label.upper()}')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Test Samples Classification')
plt.legend()
plt.show()

# Changing C value and adding new training sample
update_plot(new_black=[4,4], C_value=1, title_suffix='with Black (4,4) & C=1')
update_plot(new_black=[4,4], C_value=1e10, title_suffix='with Black (4,4) & C=∞')
