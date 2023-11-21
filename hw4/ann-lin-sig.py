import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the neural network structure
class SimpleNeuralNetwork:
    def __init__(self):
        # Initialize weights
        # Weights for layer 1 (hidden layer)
        self.weights_A_C = -1
        self.weights_B_C = 0
        self.weights_A_D = 2
        self.weights_B_D = 1
        self.weights_A_E = -2
        self.weights_B_E = 1

        # Weights for layer 2 (output layer)
        self.weights_C_F = 2
        self.weights_D_F = -0.5
        self.weights_E_F = 1
        self.weights_C_G = -2
        self.weights_D_G = 1
        self.weights_E_G = 0.5

    def forward_pass(self, input_A, input_B):
        # Calculate outputs for the hidden layer (linear)
        output_C_linear = (input_A * self.weights_A_C) + (input_B * self.weights_B_C)
        output_D_linear = (input_A * self.weights_A_D) + (input_B * self.weights_B_D)
        output_E_linear = (input_A * self.weights_A_E) + (input_B * self.weights_B_E)

        print(f"Linear Output C: {output_C_linear} = ({input_A} * {self.weights_A_C}) + ({input_B} * {self.weights_B_C})")
        print(f"Linear Output D: {output_D_linear} = ({input_A} * {self.weights_A_D}) + ({input_B} * {self.weights_B_D})")
        print(f"Linear Output E: {output_E_linear} = ({input_A} * {self.weights_A_E}) + ({input_B} * {self.weights_B_E})")

        # Calculate outputs for the output layer (linear)
        output_F_linear = (output_C_linear * self.weights_C_F) + \
                          (output_D_linear * self.weights_D_F) + \
                          (output_E_linear * self.weights_E_F)

        output_G_linear = (output_C_linear * self.weights_C_G) + \
                          (output_D_linear * self.weights_D_G) + \
                          (output_E_linear * self.weights_E_G)

        print(f"Linear Output F: {output_F_linear} = ({output_C_linear} * {self.weights_C_F}) + ({output_D_linear} * {self.weights_D_F}) + ({output_E_linear} * {self.weights_E_F})")
        print(f"Linear Output G: {output_G_linear} = ({output_C_linear} * {self.weights_C_G}) + ({output_D_linear} * {self.weights_D_G}) + ({output_E_linear} * {self.weights_E_G})")

        # Apply sigmoid activation function
        activated_C = sigmoid(output_C_linear)
        activated_D = sigmoid(output_D_linear)
        activated_E = sigmoid(output_E_linear)

        print(f"Sigmoid Activated C: {activated_C} = sigmoid({output_C_linear})")
        print(f"Sigmoid Activated D: {activated_D} = sigmoid({output_D_linear})")
        print(f"Sigmoid Activated E: {activated_E} = sigmoid({output_E_linear})")

        # Calculate activated outputs for the output layer
        activated_F = sigmoid((activated_C * self.weights_C_F) + \
                              (activated_D * self.weights_D_F) + \
                              (activated_E * self.weights_E_F))
                              
        activated_G = sigmoid((activated_C * self.weights_C_G) + \
                              (activated_D * self.weights_D_G) + \
                              (activated_E * self.weights_E_G))

        print(f"Sigmoid Activated F: {activated_F} = sigmoid(({activated_C} * {self.weights_C_F}) + ({activated_D} * {self.weights_D_F}) + ({activated_E} * {self.weights_E_F}))")
        print(f"Sigmoid Activated G: {activated_G} = sigmoid(({activated_C} * {self.weights_C_G}) + ({activated_D} * {self.weights_D_G}) + ({activated_E} * {self.weights_E_G}))")

        return {
            "linear": {"F": output_F_linear, "G": output_G_linear},
            "sigmoid": {"F": activated_F, "G": activated_G}
        }

# Create an instance of the neural network
network = SimpleNeuralNetwork()

# Function to get input and produce output
def get_output(input_A, input_B):
    return network.forward_pass(input_A, input_B)

# Example usage:
input_A = float(input("Enter input for A: "))
input_B = float(input("Enter input for B: "))
outputs = get_output(input_A, input_B)

print("Linear Output:", outputs['linear'])
print("Sigmoid Output:", outputs['sigmoid'])
