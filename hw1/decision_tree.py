#-------------------------------------------------------------------------
# AUTHOR: Jason Jones
# FILENAME: decision_tree.py
# SPECIFICATION: This program reads a csv file and creates a decision tree based on the data
# FOR: CS 4210- Assignment #1
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

# transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
#--> add your Python code here
X = [] 
for i in range(len(db)):
    X.append([])
    for j in range(4):
        if db[i][j] == 'Young' or db[i][j] == 'Myope' or db[i][j] == 'Yes' or db[i][j] == 'Reduced':
            X[i].append(1)
        elif db[i][j] == 'Prepresbyopic' or db[i][j] == 'Hypermetrope' or db[i][j] == 'No' or db[i][j] == 'Normal':
            X[i].append(2)
        elif db[i][j] == 'Presbyopic' or db[i][j] == 'No':
            X[i].append(3)
        else:
            X[i].append(4)
            

 


#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> addd your Python code here
Y = [] 
for i in range(len(db)): 
    if db[i][4] == 'Yes':
        Y.append(1)
    else:
        Y.append(2)


#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()