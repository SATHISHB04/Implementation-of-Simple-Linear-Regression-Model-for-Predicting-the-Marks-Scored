# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
print("df.head")

df.head()

print("df.tail")

df.tail()

Y=df.iloc[:,1].values
print("Array of Y")
Y

X=df.iloc[:,:-1].values
print("Array of X")
X

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

print("Values of Y prediction")
Y_pred

print("Array values of Y test")
Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours Vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
print("Training Set Graph")
plt.show()

plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours Vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
print("Test Set Graph")
plt.show()

print("Values of MSE, MAE and RMSE")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: SATHISH.B

RegisterNumber:212224040299

## Output:
<img width="1920" height="1200" alt="Screenshot (4)" src="https://github.com/user-attachments/assets/8bd27822-cd57-4dfa-be04-e6409a4b1a2c" />
<img width="1920" height="1200" alt="Screenshot (3)" src="https://github.com/user-attachments/assets/8b985ef8-df89-4313-b4a6-01d090d08826" />
<img width="1920" height="1200" alt="Screenshot (2)" src="https://github.com/user-attachments/assets/1556d45b-75c5-4b56-b3fb-1a159788ebe6" />
<img width="1920" height="1200" alt="Screenshot (1)" src="https://github.com/user-attachments/assets/18d4d142-68d8-43dd-aaac-b656cbacf7f2" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
