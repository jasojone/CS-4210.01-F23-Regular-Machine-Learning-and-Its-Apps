#-------------------------------------------------------------------------
# AUTHOR: Jason Jones
# FILENAME: decision_tree_2.py
# SPECIFICATION: 
# This program reads in 3 different training sets and trains a decision tree on each one. 
# It then tests the decision tree on the same test set for each training set and prints the accuracy for each run and the average accuracy for each training set.
# FOR: CS 4210- Assignment #2
# TIME SPENT: My entire life up to this point. 8 hours on this assignment due to the heavy interruptions from my family.
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append(row)

    # transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # Converts the features in the data set into numeric values.
    # The data set is a set of features of patients who have a certain type of eye condition.
    # The features are age, spectacle prescription, astigmatism, and tear production rate.
    # The values for the features are Young, Prepresbyopic, Presbyopic, Hypermetrope, Myope,
    # No, Yes, Normal, and Reduced. The code converts these values into numbers so that they
    # can be used in a machine learning algorithm.

    age_feature = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
    spectacle_feature = {"Hypermetrope": 1, "Myope": 2}
    astigmatism_feature = {"No": 1, "Yes": 2}
    tear_feature = {"Normal": 1, "Reduced": 2}
    
    # Converts the string values to numerical values
    # It then appends the values to the X and Y lists
    # X contains the features, Y contains the labels

    for row in dbTraining:
        row[0] = age_feature[row[0]]
        row[1] = spectacle_feature[row[1]]
        row[2] = astigmatism_feature[row[2]]
        row[3] = tear_feature[row[3]]
        X.append(row[:4])
        Y.append(row[4])
        # print(row)
    # print(X)
    # print(Y)

    #transform the original categorical training classes to numbers and add to the vector Y. 
    # Converts the Yes/No values in the Y array into 1/2 values.
    class_mapping = {"Yes": 1, "No": 2} 
    for i in range(len(Y)): 
        Y[i] = class_mapping.get(Y[i], Y[i])
    # print(Y) 
    total_accuracy = 0 
    
    #loop your training and test tasks 10 times here
    for j in range (10):
        #fitting the decision tree to the data setting max_depth=3 and entropy as the criteria for choosing the best attribute (i.e., information gain)
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)
        #read the test data and add this data to dbTest
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
           reader = csv.reader(csvfile)
           for i, row in enumerate(reader):
               if i > 0: #skipping the header
                dbTest.append (row)
        
        correct_predictions = 0
        #transform the features of the test instances to numbers following the same strategy done during training,
        # Replaces features with their corresponding values in the feature dictionaries
        # If the feature is not in the dictionary, it stays the same

        for data in dbTest:
            data[0] = age_feature.get(data[0], data[0])
            data[1] = spectacle_feature.get(data[1], data[1])
            data[2] = astigmatism_feature.get(data[2], data[2])
            data[3] = tear_feature.get(data[3], data[3])
            
            # compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            # Predicts the class of the data and compares it to the actual class. 
            # If the prediction is correct, we increment the correct_predictions variable.
            class_predicted = clf.predict([data[:4]])[0]
            true_label = class_mapping.get(data[4], data[4])
            if class_predicted == true_label:
                correct_predictions += 1
        #find the average of this model during the 10 runs (training and test set)
        accuracy = correct_predictions/len(dbTest)
        total_accuracy += accuracy
        print(f"Accuracy for run {j+1}: {accuracy:.2f}")
    #print the average accuracy of this model during the 10 runs (training and test set).
    average_accuracy = total_accuracy/10
    print(f"Average accuracy when training on {ds}: {average_accuracy:.2f}")
    print("-------------------------------------------------------------------")
    
