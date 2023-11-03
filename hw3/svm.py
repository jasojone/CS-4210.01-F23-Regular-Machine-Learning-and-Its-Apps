#-------------------------------------------------------------------------
# AUTHOR: Jason Jones
# FILENAME: svm.py
# SPECIFICATION: This program reads in a training and test set of data and uses SVM to classify the data.
# FOR: CS 4210- Assignment #3
# TIME SPENT: 1.5 hours 
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

# Defining the hyperparameter values
c = [0.1, 0.5, 1, 3, 5, 10]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf", "sigmoid"]
decision_function_shape = ["ovo", "ovr"]
gamma = [0.01]
# Reading the data by using Pandas library
df = pd.read_csv('hw3/optdigits.tra', sep=',', header=None)

# Converting the training dataframe into a numpy array for the first 64 fields/features (intensity values of the digits)
X_training = np.array(df.values)[:,:64]

# Storing the values of the last column/feature (digits)
y_training = np.array(df.values)[:,-1]

# Reading the data by using Pandas library
df = pd.read_csv('hw3/optdigits.tes', sep=',', header=None)

# Converting the testing dataframe into a numpy array for the first 64 fields/features (intensity values of the digits)
X_test = np.array(df.values)[:,:64]

# Storing the values of the last column/feature (digits)
y_test = np.array(df.values)[:,-1]

# Initializing the highest accuracy
highest_accuracy = 0
best_params = {}

# Open a file to write the accuracies
with open('svm_accuracies.txt', 'w') as accuracy_file:
    # Write the table header for all results
    accuracy_file.write(f"| {'C':^10} | {'Degree':^10} | {'Kernel':^10} | {'Gamma':^10} | {'Decision Function':^20} | {'Accuracy':^10} |\n")
    accuracy_file.write("-" * 75 + "\n")

    # Iterating through the hyperparameters
    for c_value in c: 
        for degree_value in degree:
            for kernel_value in kernel:
                # Only use gamma if the kernel is 'rbf', 'poly', or 'sigmoid'
                if kernel_value in ['rbf', 'poly', 'sigmoid']:
                    gamma_values = gamma
                else:
                    gamma_values = ['scale']  # Default value for linear kernel
                    
                for gamma_value in gamma_values:
                    for decision_function_value in decision_function_shape:
                        # Creating the SVM classifier
                        clf = svm.SVC(C=c_value, degree=degree_value, kernel=kernel_value, gamma=gamma_value, decision_function_shape=decision_function_value)

                        # Fitting the classifier to the training data
                        clf.fit(X_training, y_training)

                        # Making predictions and computing accuracy
                        correct_predictions = 0
                        for (x_testSample, y_testSample) in zip(X_test, y_test): # 
                            prediction = clf.predict([x_testSample])
                            if prediction == y_testSample:
                                correct_predictions += 1
                        accuracy = correct_predictions / len(y_test)

                        # Write current parameters and their resulting accuracy to the file
                        accuracy_file.write(f"| {c_value:^10} | {degree_value:^10} | {kernel_value:^10} | {gamma_value:^10} | {decision_function_value:^20} | {accuracy:^10.4f} |\n")

                        # Checking if the current accuracy is higher than the previously highest accuracy
                        if accuracy > highest_accuracy:
                            highest_accuracy = accuracy
                            best_params = {'C': c_value, 'degree': degree_value, 'kernel': kernel_value, 'gamma': gamma_value, 'shape': decision_function_value}
# Print the best parameters and the highest accuracy in a tabular format
print("-" * 75)
print(f"| {'Best Parameters':^20} | {'Value':^20} |")
print("-" * 45)
for param, value in best_params.items():
    print(f"| {param:^20} | {value:^20} |")
print("-" * 45)
print(f"| {'Highest Accuracy':^20} | {highest_accuracy:^20.4f} |")
print("-" * 45)
