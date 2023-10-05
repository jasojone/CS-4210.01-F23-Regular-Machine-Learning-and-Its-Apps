#-------------------------------------------------------------------------
# AUTHOR: Jason Jones
# FILENAME: naive_bayes.py
# SPECIFICATION: 
# This program reads in a training set and a test set and trains a naive bayes classifier on the training set.
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB

#reading the training data in a csv file
#--> add your Python code here
import csv
training_data = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            training_data.append(row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
outlook_feature = {"Sunny": 1, "Overcast": 2, "Rain": 3}
temperature_feature = {"Hot": 1, "Mild": 2, "Cool": 3}
humidity_feature = {"High": 1, "Normal": 2}
wind_feature = {"Weak": 1, "Strong": 2}
X = []

for row in training_data:
    X.append([outlook_feature[row[1]], temperature_feature[row[2]], humidity_feature[row[3]], wind_feature[row[4]]])

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
class_mapping = {"Yes": 1, "No": 2}
Y = [class_mapping[row[5]] for row in training_data]
# print(Y)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
test_data = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            test_data.append(row)

#printing the header os the solution
print("The predictions with probabilities greater than 0.75 of weather_test.csv using the naive bayes classifier are:")

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
print("Sample    Outlook    Temperature    Humidity    Windy    Play    Probability")
for row in test_data:
    test_sample = [outlook_feature[row[1]], temperature_feature[row[2]], humidity_feature[row[3]], wind_feature[row[4]]]
    probability = clf.predict_proba([test_sample])[0]
    if probability[0] >= 0.75:
        print(f"{row[0]:<10}{row[1]:<12}{row[2]:<15}{row[3]:<12}{row[4]:<10}Yes      {probability[0]:.2f}")
    elif probability[1] >= 0.75:
        print(f"{row[0]:<10}{row[1]:<12}{row[2]:<15}{row[3]:<12}{row[4]:<10}No       {probability[1]:.2f}")



