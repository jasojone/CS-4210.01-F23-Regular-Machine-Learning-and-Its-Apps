import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

class SVMVisualization:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.clf = self.train_svm(C_value=1e6)

    def train_svm(self, C_value):
        clf = svm.SVC(kernel='linear', C=C_value)
        clf.fit(self.data, self.labels)
        return clf

    def update_data(self, new_data, new_label):
        self.data = np.vstack((self.data, new_data))
        self.labels = np.append(self.labels, new_label)
        self.clf = self.train_svm(C_value=1e6)

    def plot_decision_boundary(self, plot_support=True, test_samples=None, title=''):
        plt.figure(figsize=(10, 7))
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, s=30, cmap=plt.cm.Paired, label='Training Points')
        if test_samples:
            for label, sample in test_samples.items():
                plt.scatter(sample['point'][0], sample['point'][1], color=sample['color'], marker='s', s=50, edgecolors='gray', label=f'Test Sample {label.upper()}')
        self.plot_svc_decision_function(self.clf, plot_support)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(title)
        plt.legend()

    def plot_svc_decision_function(self, model, plot_support):
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = model.decision_function(xy).reshape(XX.shape)
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        if plot_support:
            ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=200, linewidth=1, facecolors='none', edgecolors='k')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

def main():
    # Original dataset
    yellow_points = np.array([[1,2], [2,1], [2,3], [4,3]])
    black_points = np.array([[6,5], [7,8], [8,3], [8,4]])
    all_points = np.vstack((yellow_points, black_points))
    labels = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    svm_viz = SVMVisualization(all_points, labels)
    svm_viz.plot_decision_boundary(plot_support=True, title='Initial Decision Boundary & Support Vectors')
    
    # Add new training samples and update plot for black (7,5)
    svm_viz.update_data(new_data=[7,5], new_label=-1)
    svm_viz.plot_decision_boundary(plot_support=True, title='Updated with Black (7,5)')
    # Add new training samples and update plot for yellow (4,2)
    svm_viz.update_data(new_data=[4,2], new_label=1)
    svm_viz.plot_decision_boundary(plot_support=True, title='Updated with Yellow (4,2)')
    
    # Test samples
    test_samples = {
        'd': {'point': [7,5], 'color': 'black'},
        'e': {'point': [6,4], 'color': 'black'},
        'f': {'point': [4,2], 'color': 'yellow'},
        'g': {'point': [5,3], 'color': 'yellow'},
        'h': {'point': [5,3], 'color': 'black'},
        'i': {'point': [6,4], 'color': 'yellow'}
    }
    svm_viz.plot_decision_boundary(plot_support=True, test_samples=test_samples, title='Test Samples Classification')
    plt.show()
    

if __name__ == "__main__":
    main()
