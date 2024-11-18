# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries: Load necessary libraries for data manipulation, encoding, model training, and evaluation.

2.Load the dataset: Read the CSV file containing the salary data.

3.Explore the dataset and Display the first few rows to understand the structure and Check the data types and null values for each column.

4.Identify and handle any missing values.

5.Encode categorical data: Use LabelEncoder to transform categorical variables like "Position" into numerical values.

6.Feature selection: Select the independent variables (features) such as "Position" and "Level".

7.Select the target variable: Assign the dependent variable, "Salary", as the target.

8.Split the dataset: Divide the dataset into training and testing sets, ensuring a portion of the data is reserved for model evaluation.

9.Initialize and train the model: Use the Decision Tree Regressor to fit the model on the training data.

10.Make predictions: Use the trained model to predict salaries for the test dataset.

11.Evaluate the model
12.Calculate the Mean Squared Error (MSE) to measure the average squared difference between actual and predicted values.

13Calculate the R-squared score to determine how well the model explains the variability in the data.

14.Make new predictions: Use the model to predict the salary for a specific combination of "Position" and "Level" values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Tanushree.A
RegisterNumber:212223100057  
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
X=data[["Position","Level"]]
Y=data[["Salary"]]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(X_train,Y_train)
Y_pred = dt.predict(X_test)
from sklearn import metrics
mse=metrics.mean_squared_error(Y_test,Y_pred)
print(mse)
r2=metrics.r2_score(Y_test,Y_pred)
print(r2)
dt.predict([[4,2]])
```

## Output:
## Mean squared error
![image](https://github.com/user-attachments/assets/c5b95cb0-3cef-4e8e-92a4-e04556ed9890)

## r2 score
![image](https://github.com/user-attachments/assets/941c1c2e-8f34-4939-afcb-f1771ff009e2)

## predicted value
![image](https://github.com/user-attachments/assets/d27d7128-8972-4b00-8a6b-2d4c0a776105)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
