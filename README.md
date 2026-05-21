# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Detect File Encoding: Use chardet to determine the dataset's encoding.
2.Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3.Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4.Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5.Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6.Train SVM Model: Fit an SVC model on the training data.
7.Predict Labels: Predict test labels using the trained SVM model.
8.Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: AJITHKUMAR J
RegisterNumber: 212225040015 
*/
```
```
Program to implement the SVM For Spam Mail Detection.
Developed by: KABELAN G K
RegisterNumber: 24900985
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect (rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv('spam.csv', encoding='Windows-1252')
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
<img width="1091" height="566" alt="image" src="https://github.com/user-attachments/assets/2263823e-1348-423d-9f9f-bd9686642163" />
<img width="798" height="620" alt="image" src="https://github.com/user-attachments/assets/e52bd1a3-bc73-48eb-8995-dfd06008846e" />
<img width="1160" height="497" alt="image" src="https://github.com/user-attachments/assets/3cde674a-2c32-489e-8847-b06391ba1f59" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
