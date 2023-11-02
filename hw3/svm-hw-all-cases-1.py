import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
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
# plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=200, linewidth=1, facecolors='none', edgecolors='k')
        for sv in model.support_vectors_:
            ax.text(sv[0], sv[1], f"({sv[0]:.2f}, {sv[1]:.2f})", ha='right', va='bottom', fontsize=8)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
# def plot_additional_decision_boundaries(ax=None):
#     if ax is None:
#         ax = plt.gca()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()

#     # Parameters for the additional decision boundaries
#     slopes = [0.8, -1.2]  # added negative slope
#     intercepts = [0.8, 6.5]  # modified intercept for negative slope
#     margins = [0.3, 0.3]

#     for slope, intercept, margin in zip(slopes, intercepts, margins):
#         xfit = np.linspace(xlim[0], xlim[1], 30)
#         yfit = slope * xfit + intercept
#         ax.plot(xfit, yfit, '-k')
#         ax.fill_between(xfit, yfit - margin, yfit + margin, edgecolor='none', color='#AAAAAA', alpha=0.4)

#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
    
def print_decision_boundary(w, b, title_suffix=''):
    # Decision boundary equation in the form w.x + b = 0
    equation = f"{title_suffix} Decision boundary equation: {w[0]:.3f}x1 + {w[1]:.3f}x2 + {b:.3f} = 0"
    plt.text(0.05, 0.65, equation, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))  
    
def train_svm(data, labels, C_value):
    # SVM training
    clf = svm.SVC(kernel='linear', C=C_value)
    clf.fit(data, labels)
    return clf

def update_plot(new_yellow=None, new_black=None, C_value=0, title_suffix=''):
    global yellow_points, black_points, all_points, labels, clf
    if new_yellow is not None:
        yellow_points = np.vstack((yellow_points, new_yellow))
        all_points = np.vstack((all_points, new_yellow))
        labels = np.append(labels, 1)
    if new_black is not None:
        black_points = np.vstack((black_points, new_black))
        all_points = np.vstack((all_points, new_black))
        labels = np.append(labels, -1)
    
    # SVM training
    clf = train_svm(all_points, labels, C_value)
    
    # Extracting the weight vector (w) and bias (b)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    
    # Plot updated decision boundary and support vectors
    plt.figure(figsize=(10, 7))
    plt.scatter(yellow_points[:, 0], yellow_points[:, 1], color='yellow', s=30, label='Yellow Points')
    plt.scatter(black_points[:, 0], black_points[:, 1], color='black', s=30, label='Black Points')
    plot_svc_decision_function(clf, plot_support=True)
    print_decision_boundary(w, b, 'Updated Plot')  # Call the function to plot the decision boundary
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'Updated Decision Boundary & Support Vectors {title_suffix}')
    plt.legend()
    # plt.show()

def initial_setup():
    global yellow_points, black_points, all_points, labels, clf
    # Original dataset
    yellow_points = np.array([[1,2], [2,1], [2,3], [4,3]])
    black_points = np.array([[6,5], [7,8], [8,3], [8,4]])
    all_points = np.vstack((yellow_points, black_points))
    labels = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    # SVM training
    clf = train_svm(all_points, labels, C_value=1e6)

def plot_initial_decision_boundary():
    plt.figure(figsize=(10, 7))
    plt.scatter(yellow_points[:, 0], yellow_points[:, 1], color='yellow', s=30, label='Yellow Points')
    plt.scatter(black_points[:, 0], black_points[:, 1], color='black', s=30, label='Black Points')
    plot_svc_decision_function(clf, plot_support=True)
    # Extracting the weight vector (w) and bias (b)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    print_decision_boundary(w, b, '')  # Call the function to plot the decision boundary
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Initial Decision Boundary & Support Vectors')
    plt.legend()
    # plt.show()
    
def plot_test_samples():
    # Test samples
    test_samples = {
        'd': {'point': [7,5], 'color': 'black'},
        'e': {'point': [6,4], 'color': 'black'},
        'f': {'point': [4,2], 'color': 'yellow'},
        'g': {'point': [5,3], 'color': 'yellow'},
        'h': {'point': [5,3], 'color': 'black'},
        'i': {'point': [6,4], 'color': 'yellow'}
    }
    
    plt.figure(figsize=(10, 7))
    plt.scatter(yellow_points[:, 0], yellow_points[:, 1], color='yellow', s=30, label='Yellow Points')
    plt.scatter(black_points[:, 0], black_points[:, 1], color='black', s=30, label='Black Points')
    plot_svc_decision_function(clf, plot_support=True)
    # Extracting the weight vector (w) and bias (b)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    print_decision_boundary(w, b)  # Call the function to plot the decision boundary
    for label, sample in test_samples.items():
        plt.scatter(sample['point'][0], sample['point'][1], color=sample['color'], marker='s', s=50, edgecolors='gray', label=f'Test Sample {label.upper()}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Test Samples Classification')
    plt.legend()
    # plt.show()

def main():
    initial_setup()
    plot_initial_decision_boundary()
    # Calculating the decision function for each support vector
    def calculate_decision_function(support_vectors, coef, intercept):
        return np.dot(support_vectors, coef) + intercept

    # Print necessary information for math calculations
    w = clf.coef_[0]
    b = clf.intercept_[0]
    support_vectors = clf.support_vectors_
    decision_function_values = calculate_decision_function(support_vectors, w, b)
    print("w: ", w)
    print("b: ", b)
    print("support vectors: ", support_vectors)
    print("decision function values for support vectors: ", decision_function_values)
    
    
    # Plotting the decision boundary
    # plt.figure(figsize=(10, 7))
    # plot_additional_decision_boundaries()  # Plotting the additional decision boundary
    # plt.show()
    
    # Add new training samples and update plot
    update_plot(new_black=[7,5], title_suffix='with Black (7,5)', C_value=1e6)
    update_plot(new_yellow=[4,2], title_suffix='with Yellow (4,2)', C_value=1e6)
    # Plot test samples
    plot_test_samples()
    # Changing C value and adding new training sample
    update_plot(new_black=[4,4], C_value=1, title_suffix='with Black (4,4) & C=1')
    w = clf.coef_[0]
    b = clf.intercept_[0]
    support_vectors = clf.support_vectors_
    decision_function_values = calculate_decision_function(support_vectors, w, b)

    print("After retraining:")
    print("w: ", w)
    print("b: ", b)
    print("support vectors: ", support_vectors)
    print("decision function values for support vectors: ", decision_function_values)
    
    update_plot(new_black=[4,4], C_value=1e6, title_suffix='with Black (4,4) & C=1e6')
    w = clf.coef_[0]
    b = clf.intercept_[0]
    support_vectors = clf.support_vectors_
    decision_function_values = calculate_decision_function(support_vectors, w, b)

    print("After retraining:")
    print("w: ", w)
    print("b: ", b)
    print("support vectors: ", support_vectors)
    print("decision function values for support vectors: ", decision_function_values)
    
    
    plt.show()

main()
