import numpy as np

# Define the perceptron function
def perceptron(inputs, weights, bias):
    # Calculate the weighted sum of inputs
    weighted_sum = np.dot(inputs, weights) + bias
    
    # Apply the activation function (step function)
    if weighted_sum >= 0:
        return 1
    else:
        return 0

# Define the AND operator
def AND(x1, x2):
    inputs = np.array([x1, x2])
    weights = np.array([1, 1])
    bias = -1.5
    return perceptron(inputs, weights, bias)

# Define the OR operator
def OR(x1, x2):
    inputs = np.array([x1, x2])
    weights = np.array([1, 1])
    bias = -0.5
    return perceptron(inputs, weights, bias)

# Define the NOT operator
def NOT(x):
    inputs = np.array([x])
    weights = np.array([-1])
    bias = 0.5
    return perceptron(inputs, weights, bias)

# Define the XOR operator
def XOR(x1, x2):
    return AND(NOT(AND(x1, x2)), OR(x1, x2))

# Define the multi-layered XOR operator
def multi_layered_XOR(x1, x2):
    h1 = XOR(x1, x2)
    h2 = XOR(x1, h1)
    h3 = XOR(x2, h1)
    output = XOR(h2, h3)
    return output
# create a table for each operator

# AND
print("AND")
print("0 0: " + str(AND(0, 0)))
print("0 1: " + str(AND(0, 1)))
print("1 0: " + str(AND(1, 0)))
print("1 1: " + str(AND(1, 1)))

# OR
print("OR")
print("0 0: " + str(OR(0, 0)))
print("0 1: " + str(OR(0, 1)))
print("1 0: " + str(OR(1, 0)))
print("1 1: " + str(OR(1, 1)))

# NOT
print("NOT")
print("0: " + str(NOT(0)))
print("1: " + str(NOT(1)))

# XOR
print("XOR")
print("0 0: " + str(XOR(0, 0)))
print("0 1: " + str(XOR(0, 1)))
print("1 0: " + str(XOR(1, 0)))
print("1 1: " + str(XOR(1, 1)))

# multi-layered XOR
print("multi-layered XOR")
print("0 0: " + str(multi_layered_XOR(0, 0)))
print("0 1: " + str(multi_layered_XOR(0, 1)))
print("1 0: " + str(multi_layered_XOR(1, 0)))
print("1 1: " + str(multi_layered_XOR(1, 1)))
