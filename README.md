Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Murali Krishna S  
RegisterNumber:212223230129
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/f57c5741-5bf7-44a7-92cb-6d40d0d4e5bb)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/1ee34ef2-f36e-4f6d-9356-e2c608133741)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/484f0a02-ac5f-4093-81f3-9ca6cec468cd)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/125d37c0-9fc4-4802-8904-c64ef4e951a4)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/1d374e1a-a60f-4354-b983-fb694b864851)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/4e5cbb2e-3ec3-44a0-b25c-cf79b9fd3fd3)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/374bb950-1513-432f-a9ba-1c83ed142c74)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/1d2ab0d1-90a8-48a1-9ced-177739c3d573)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/861f9bd7-e913-4ef5-8887-ed1261a9e8d8)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
