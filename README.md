# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset and split into training and testing sets.

2. Train the Linear Regression model using training data.

3. Predict the results using testing data.

4. Evaluate model performance using error metrics and visualize results.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: THAMIZH SELVAN R
RegisterNumber:  212222230158
*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```
## Output:

### Head Values
<img width="202" height="127" alt="Screenshot 2025-09-09 155352" src="https://github.com/user-attachments/assets/5fdfe1fa-4141-4d63-9764-06dbeece1412" />

### Tail Values
<img width="198" height="131" alt="Screenshot 2025-09-09 155403" src="https://github.com/user-attachments/assets/c8918054-bdd4-48e7-8701-4c9bfa05c197" />

### Compare Dataset
<img width="802" height="585" alt="image" src="https://github.com/user-attachments/assets/64b0445c-7e7d-4ac0-b9ef-8e5d898bff5f" />


### Predication values of X and Y
<img width="786" height="78" alt="image" src="https://github.com/user-attachments/assets/aee14b46-8d2f-4a39-8c72-69d41ad59fa1" />

### Training set
<img width="775" height="576" alt="Screenshot 2025-09-09 155502" src="https://github.com/user-attachments/assets/98b72660-08c4-41fb-97b4-5aabd9b41965" />

### Testing Set
<img width="766" height="575" alt="Screenshot 2025-09-09 155530" src="https://github.com/user-attachments/assets/60aafcf9-792a-49ca-bccf-5eff43754b97" />

### MSE,MAE and RMSE
<img width="468" height="85" alt="Screenshot 2025-09-09 155543" src="https://github.com/user-attachments/assets/7ec8cab8-159e-4d77-89e8-932945350b6b" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
