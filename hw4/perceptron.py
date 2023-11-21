#-------------------------------------------------------------------------
# AUTHOR: Jason Jones
# FILENAME: perceptron.py
# SPECIFICATION: Build and compare Single Layer and Multi-Layer Perceptron classifiers
# FOR: CS 4210- Assignment #4
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# Importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

# Hyperparameters
n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]  # Learning rates
r = [True, False]  # Shuffle options

# Reading the data by using Pandas library
df = pd.read_csv('hw4/optdigits.tra', sep=',', header=None)

# Getting the first 64 fields to form the feature data for training
X_training = np.array(df.values)[:,:64]
# Getting the last field to form the class label for training
y_training = np.array(df.values)[:,-1]

# Reading the data by using Pandas library
df = pd.read_csv('hw4/optdigits.tes', sep=',', header=None)

# Getting the first 64 fields to form the feature data for test
X_test = np.array(df.values)[:,:64]
# Getting the last field to form the class label for test
y_test = np.array(df.values)[:,-1]

# Initialize variables to track the best accuracy and corresponding hyperparameters
best_accuracy_perceptron, best_params_perceptron = 0, {}
best_accuracy_mlp, best_params_mlp = 0, {}

for learning_rate in n:

    for shuffle in r:

        for algorithm in ['Perceptron', 'MLP']:

            # Create a Neural Network classifier
            if algorithm == 'Perceptron':
                clf = Perceptron(eta0=learning_rate, shuffle=shuffle, max_iter=1000)
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init=learning_rate, hidden_layer_sizes=(100,), shuffle=shuffle, max_iter=1000)

            # Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            # Make the classifier prediction for each test sample and start computing its accuracy
            correct_predictions = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])[0]
                if prediction == y_testSample:
                    correct_predictions += 1
            accuracy = correct_predictions / len(y_test)

            # Check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            # and print it together with the network hyperparameters
            # Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            # Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            if algorithm == 'Perceptron' and accuracy > best_accuracy_perceptron:
                best_accuracy_perceptron = accuracy
                best_params_perceptron = {'learning_rate': learning_rate, 'shuffle': shuffle}
                print(f"Highest Perceptron accuracy so far: {best_accuracy_perceptron}, Parameters: {best_params_perceptron}")

            if algorithm == 'MLP' and accuracy > best_accuracy_mlp:
                best_accuracy_mlp = accuracy
                best_params_mlp = {'learning_rate': learning_rate, 'shuffle': shuffle}
                print(f"Highest MLP accuracy so far: {best_accuracy_mlp}, Parameters: {best_params_mlp}")
