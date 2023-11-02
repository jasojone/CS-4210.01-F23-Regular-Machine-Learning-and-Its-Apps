import numpy as np
from sklearn import svm

# Function to train SVM
def train_svm(data, labels, C_value):
    clf = svm.SVC(kernel='linear', C=C_value)
    clf.fit(data, labels)
    return clf

# Main function to solve the problems
def solve_svm_problems():
    # One-dimensional data
    points = np.array([-3, 0, 1, 2, 3, 4, 5]).reshape(-1, 1)
    labels = np.array([-1, -1, 1, 1, 1, 1, 1])

    # Train SVM with high C value (hard margin)
    clf = train_svm(points, labels, C_value=1e6)

    # Support Vectors
    support_vectors = clf.support_vectors_
    print(f"Support Vectors (x coordinate): {support_vectors.flatten()}")

    # Solution parameters w and b
    w = clf.coef_[0]
    b = clf.intercept_[0]
    print(f"w: {w[0]}, b: {b}")

    # Width of the margin
    margin_width = 1 / np.linalg.norm(w)
    print(f"Width of the margin: {margin_width}")

    # SVM classifications for the test data {-1.5, 1.5}
    test_points = np.array([-1.5, 1.5]).reshape(-1, 1)
    test_labels = clf.predict(test_points)
    print(f"SVM classifications for {-1.5}: {test_labels[0]}, for {1.5}: {test_labels[1]}")

    # Remove the point (1,+) and train the SVM again
    points_new = np.array([-3, 0, 2, 3, 4, 5]).reshape(-1, 1)
    labels_new = np.array([-1, -1, 1, 1, 1, 1])
    clf_new = train_svm(points_new, labels_new, C_value=1e6)

    # New solution parameters w and b
    w_new = clf_new.coef_[0]
    b_new = clf_new.intercept_[0]
    print(f"New w: {w_new[0]}, New b: {b_new}")

    # New width of the margin
    margin_width_new = 1 / np.linalg.norm(w_new)
    print(f"New width of the margin: {margin_width_new}")

solve_svm_problems()
