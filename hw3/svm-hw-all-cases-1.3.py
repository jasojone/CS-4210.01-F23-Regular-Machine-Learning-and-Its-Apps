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
        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], 
                   s=200, linewidth=1, facecolors='none', edgecolors='k')
        # Label support vectors
        for i, support_vector in enumerate(sv):
            ax.annotate(f'SV{i+1}: ({support_vector[0]}, {support_vector[1]})', (support_vector[0], support_vector[1]), textcoords="offset points", xytext=(0,10), ha='center')

    # Plot hyperplane equation (for linear kernel)
    if model.kernel == 'linear':
        w = model.coef_[0]
        b = model.intercept_[0]
        x = np.linspace(xlim[0], xlim[1], 200)
        y = -(w[0]/w[1])*x - b/w[1]
        ax.plot(x, y, 'k-', lw=1)
        mid_point = (xlim[0] + xlim[1]) / 2
        margin = 1 / np.linalg.norm(w)
        ax.annotate(f'{w[0]:.2f}x1 + {w[1]:.2f}x2 + {b:.2f} = 0\nMargin: {margin:.2f}', xy=(mid_point, -(w[0]/w[1])*mid_point - b/w[1]), textcoords="offset points", xytext=(0,10), ha='center')
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Function to train SVM
def train_svm(data, labels, C_value):
    clf = svm.SVC(kernel='linear', C=C_value)
    clf.fit(data, labels)
    return clf

# Function to plot decision boundary and support vectors
def plot_decision_boundary(clf, points, labels, title, new_point=None, new_label=None, test_samples=None):
    plt.figure(figsize=(10, 7))
    
    # Plot data points: yellow for label 1, black for label -1
    plt.scatter(points[labels == 1, 0], points[labels == 1, 1], s=30, color='yellow', label='Class 1')
    plt.scatter(points[labels == -1, 0], points[labels == -1, 1], s=30, color='black', label='Class -1')
    
    if new_point is not None:
        new_point_color = 'yellow' if new_label == 1 else 'black'
        plt.scatter(new_point[0], new_point[1], s=50, color=new_point_color, 
                    label='New Point' if new_label is None else f'New Point (Class {new_label})')
        if new_label is not None:
            points = np.vstack((points, new_point))
            labels = np.append(labels, new_label)
    
    # Plot and classify test samples
    if test_samples is not None:
        for sample in test_samples.values():
            test_point = np.array(sample['point']).reshape(1, -1)
            test_label = clf.predict(test_point)
            test_point_color = 'yellow' if test_label == 1 else 'black'
            plt.scatter(test_point[:, 0], test_point[:, 1], s=50, 
                        color=test_point_color, label=f'Test Point {test_point} (Class {test_label})')

    plot_svc_decision_function(clf)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(title)
    # Place legend outside the plot area
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0.)
    # Print relevant data to the console
    print(f"Title: {title}")
    print(f"Support Vectors:\n{clf.support_vectors_}")
    print(f"Decision Function Weights (for linear kernel): {clf.coef_ if hasattr(clf, 'coef_') else 'N/A'}")
    print(f"w.x + b = 0 (for linear kernel): {clf.coef_[0][0]}x1 + {clf.coef_[0][1]}x2 + {clf.intercept_[0]} = 0")
    print(f"Margin: {1 / np.linalg.norm(clf.coef_)}")
    print("Data Points (x, y, label):")
    for point, label in zip(points, labels):
        print(f"({point[0]}, {point[1]}): {label}")
    if new_point is not None and new_label is not None:
        print(f"New Point (x, y, label): ({new_point[0]}, {new_point[1]}): {new_label}")
    if test_samples:
        for sample in test_samples.values():
            test_point = sample['point']
            test_label = clf.predict([test_point])
            print(f"Test Point (x, y, predicted label): ({test_point[0]}, {test_point[1]}): {test_label[0]}")
    print()
# Main function
def main():
    # Initial data setup
    points = np.array([[1,2], [2,1], [2,3], [4,3], [6,5], [7,8], [8,3], [8,4]])
    labels = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    
    # Train SVM with high C value (hard margin)
    clf = train_svm(points, labels, C_value=1e6)
    plot_decision_boundary(clf, points, labels, 'Initial Decision Boundary')
    # Add new point (7,5) with label -1 (black)
    new_point = np.array([7,5])
    new_label = -1
    clf = train_svm(np.vstack((points, new_point)), np.append(labels, new_label), C_value=1e6)
    plot_decision_boundary(clf, points, labels, 'Decision Boundary with Black (7,5)', new_point, new_label)
    # Add new point (4,2) with label 1 (yellow)
    new_point = np.array([4,2])
    new_label = 1
    clf = train_svm(np.vstack((points, new_point)), np.append(labels, new_label), C_value=1e6)
    plot_decision_boundary(clf, points, labels, 'Decision Boundary with Yellow (4,2)', new_point, new_label)
    # Classify test samples
    test_samples = {
        'd': {'point': [7,5], 'color': 'black'},
        'e': {'point': [6,4], 'color': 'black'},
        'f': {'point': [4,2], 'color': 'yellow'},
        'g': {'point': [5,3], 'color': 'yellow'},
        'h': {'point': [5,3], 'color': 'black'},
        'i': {'point': [6,4], 'color': 'yellow'}
    }
    # Train and plot with test samples
    clf = train_svm(points, labels, C_value=1e6)
    plot_decision_boundary(clf, points, labels, 'Decision Boundary with Test Samples', test_samples=test_samples)
    
    # Test case: new point (4,4) with label -1 (black) and C=1
    new_point = np.array([4,4])
    new_label = -1
    clf = train_svm(np.vstack((points, new_point)), np.append(labels, new_label), C_value=1)
    plot_decision_boundary(clf, points, labels, 'Decision Boundary with Black (4,4) & C=1', new_point, new_label)
    
    # Test case: new point (4,4) with label -1 (black) and C=1e6
    clf = train_svm(np.vstack((points, new_point)), np.append(labels, new_label), C_value=1e6)
    plot_decision_boundary(clf, points, labels, 'Decision Boundary with Black (4,4) & C=1e6', new_point, new_label)
    
    plt.show()

main()
