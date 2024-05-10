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
plt.title("Hours vs Scores(![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/8f082992-f1d2-4db0-8a7e-4404d5864276)
Training set)")
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
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/049ac67d-661d-4ca8-8380-09a9931022e2)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/580b6674-cd0e-4b50-b059-51fdc7fab9f3)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/30558100-c922-4e47-9733-fe2e885ccbb5)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/8b0dedb4-ff83-4ebc-af09-38258e5f143b)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/64cc2161-3329-4b2d-9739-9cf7d739cd03)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/9add5b59-0846-4751-82b6-befa0cd6b18c)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/4a7a2271-c4b2-414b-8e19-80742b4df57d)
![image](https://github.com/Murali-Krishna0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149054535/827b7106-0c46-403e-913d-574c59536887)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
