import numpy as np

def phi(x):
    """
    Maps x to a sixth-dimensional feature space.
    """
    x1, x2 = x
    return np.array([
        x1**2,
        np.sqrt(2) * x1 * x2,
        x2**2,
        np.sqrt(2) * x1,
        np.sqrt(2) * x2,
        1
    ])

def quadratic_kernel(x, y):
    """
    Computes the quadratic kernel for x and y.
    """
    return (np.dot(x, y) + 1) ** 2

A = (1, 2)
B = (2, 4)

# Compute phi transformations
phi_A = phi(A)
phi_B = phi(B)

# Compute dot product of phi transformations
dot_product = np.dot(phi_A, phi_B)

# Compute kernel value
kernel_value = quadratic_kernel(A, B)

# Print results with two decimal places
print("phi(A):", ["{:.2f}".format(i) for i in phi_A])
print("phi(B):", ["{:.2f}".format(i) for i in phi_B])
print("phi(A) dot phi(B): {:.2f}".format(dot_product))
print("K(A, B): {:.2f}".format(kernel_value))
