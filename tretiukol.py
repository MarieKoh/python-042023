import pandas
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus
from pydotplus import graph_from_dot_data
from sklearn.preprocessing import StandardScaler
import numpy as np
import graphviz 
from PIL import Image
from sklearn.metrics import confusion_matrix 

import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

data = pandas.read_csv("bodyPerformance.csv")
print(data.head())

X = data.drop(['class', 'sit and bend forward_cm', 'sit-ups counts', 'broad jump_cm'], axis=1)
y = data['class']

encoder = OneHotEncoder()
X = encoder.fit_transform(X)
X = X.toarray()
X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree.png')
accuracy = accuracy_score(y_test, y_pred)
image = Image.open('tree.png')
image.show()

y = data["class"]
categorical_columns = ["gender"]
numeric_columns = ["age", "height_cm", "weight_kg", "body fat_%", "diastolic", "systolic", "gripForce"]
encoded_columns = encoder.fit_transform(data[categorical_columns]).toarray() 
y_pred_decision_tree = clf.predict(X_test)

# Výpočet matice.
confusion_matrix_decision_tree = confusion_matrix(y_test, y_pred_decision_tree)
correct_class_A = confusion_matrix_decision_tree[0, 0]
classified_B = confusion_matrix_decision_tree[1, 0]
classified_C = confusion_matrix_decision_tree[2, 0]
classified_D = confusion_matrix_decision_tree[3, 0]
print('Confusion Matrix (Decision Tree):')
print(confusion_matrix_decision_tree)
print('Number of individuals correctly classified in class A:', correct_class_A)
print('Number of individuals classified into class B:', classified_B)
print('Number of individuals classified into class C:', classified_C)
print('Number of individuals classified into class D:', classified_D)

X = np.concatenate([encoded_columns, data[numeric_columns].values], axis=1)

confusion_mat = confusion_matrix(y_test, y_pred)
correct_class_A = confusion_mat[0, 0]
classified_B = confusion_mat[1, 0]
classified_C = confusion_mat[2, 0]
classified_D = confusion_mat[3, 0]
print("Confusion Matrix:")
print(confusion_mat)
print("Number of individuals correctly classified in class A:", correct_class_A)
print("Number of individuals classified into class B:", classified_B)
print("Number of individuals classified into class C:", classified_C)
print("Number of individuals classified into class D:", classified_D)

accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
y_pred_algorithm = clf.predict(X_test)
accuracy_algorithm = accuracy_score(y_test, y_pred_algorithm)
print('Accuracy (Decision Tree):', accuracy_decision_tree)
print('Accuracy (Selected Algorithm):', accuracy_algorithm)
# Algoritmus s vyšší přesností (accuracy) si vedl lépe.
