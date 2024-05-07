# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1: Start

step 2: Import Libraries: Import the necessary libraries - pandas, numpy, and matplotlib.pyplot.

step 3: Load Dataset: Load the dataset using pd.read_csv.

step 4: Remove irrelevant columns (sl_no, salary).

step 5: Convert categorical variables to numerical using cat.codes.

step 6: Separate features (X) and target variable (Y).

step 7: Define Sigmoid Function

step 8: Define the loss function for logistic regression.

step 9: Define Gradient Descent Function: Implement the gradient descent algorithm to optimize the parameters.

step 10: Training Model: Initialize theta with random values, then perform gradient descent to minimize the loss and obtain the optimal parameters.

step 11: Define Prediction Function: Implement a function to predict the output based on the learned parameters.

step 12: Evaluate Accuracy: Calculate the accuracy of the model on the training data.

step 13: Predict placement status for a new student with given feature values (xnew).

step 14: Print Results: Print the predictions and the actual values (Y) for comparison.

step 15: Stop.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: A.ARUVI.
RegisterNumber:212222230014.  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Placement_Data.csv")
data
data= data.drop('sl_no',axis=1)
data= data.drop('salary',axis=1)

data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data.dtypes
# labelling the columns
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values
Y
#initialize the mdel parameters.

theta= np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
  return 1/(1+ np.exp(-z))
def loss(theta, X,y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h) + (1-y)* np.log(1-h))
def gradient_descent (theta, X,y,alpha, num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(X.dot(theta))
    gradient=X.T.dot(h-y)/m
    theta -= alpha*gradient
  return theta
theta= gradient_descent(theta, X,y ,alpha=0.01, num_iterations=1000)
def predict(theta, X):
  h= sigmoid(X.dot(theta))
  y_pred = np.where(h>=0.5,1,0)
  return y_pred
y_pred = predict(theta, X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)
print(Y)
xnew = np.array([[ 0, 87, 0, 95, 0, 2, 78, 2, 0, 0 ,1, 0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
xnew = np.array([[ 0, 0, 0, 0, 0, 2, 8, 2, 0, 0 ,1, 0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```

## Output:
### dataset:

![image](https://github.com/Anandanaruvi/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120443233/3cb7b0a7-24a0-4a7d-a01a-b4844679f960)

### dataset.types:
+![image](https://github.com/Anandanaruvi/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120443233/31fa1d99-7680-451e-b362-d0cdee5023bd)
### dataset:

![image](https://github.com/Anandanaruvi/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120443233/0b65f19d-fbad-4d2e-b490-fb6462d1b951)

### Y:

![image](https://github.com/Anandanaruvi/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120443233/3cc98c64-401b-4360-8dd6-fedbee5cba8f)

### ACCURACY:
![image](https://github.com/Anandanaruvi/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120443233/96c38ed0-b145-4782-aa1d-92fdd7b19907)

### Y_pred:
![image](https://github.com/Anandanaruvi/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120443233/1d9e70ab-d698-498b-9ec1-cbf8c0e1dfdf)

### y:

![image](https://github.com/Anandanaruvi/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120443233/173f70aa-2652-427b-80f6-c977c1f2eb98)

### y_prednew:
![image](https://github.com/Anandanaruvi/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120443233/c3864814-c930-4548-bf30-071d8c507ad2)

![Uploading image.png…]()

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

