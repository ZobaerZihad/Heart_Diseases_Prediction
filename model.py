import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import plotly as plot
import plotly.express as px
import plotly.graph_objs as go

import cufflinks as cf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot



heart = pd.read_csv("heart.csv")


heart_df = heart.copy()



print(heart_df.head())


x= heart_df.drop(columns= 'target')
y= heart_df.target

heart.hist(figsize=(14,14))
plt.show()


numeric_columns=['trestbps','chol','thalach','age','oldpeak']
sns.pairplot(heart[numeric_columns])
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.show()

plt.figure(figsize=(12,10))
plt.subplot(221)
sns.distplot(heart[heart['target']==0].age)
plt.title('Age of patients without heart disease')
plt.subplot(222)
sns.distplot(heart[heart['target']==1].age)
plt.title('Age of patients with heart disease')
plt.subplot(223)
sns.distplot(heart[heart['target']==0].thalach )
plt.title('Max heart rate of patients without heart disease')
plt.subplot(224)
sns.distplot(heart[heart['target']==1].thalach )
plt.title('Max heart rate of patients with heart disease')
plt.show()


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=42)

log = LogisticRegression()
log.fit(x_train,y_train)
y_pred= log.predict(x_test)
p = log.score(x_test,y_test)
print("LogisticRegression:")
print(p)
print('Classification Report\n', classification_report(y_test, y_pred))
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))
cm = confusion_matrix(y_test, y_pred)
print(cm)

scaler= StandardScaler()
x_train_scaler= scaler.fit_transform(x_train)
x_test_scaler= scaler.fit_transform(x_test)

dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred= dt.predict(x_test)
p = dt.score(x_test,y_test)
print("DecisionTreeClassifier:")
print(p)
print('Classification Report\n', classification_report(y_test, y_pred))
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))
cm = confusion_matrix(y_test, y_pred)
print(cm)

RFC=RandomForestClassifier(n_estimators=20)
RFC.fit(x_train_scaler, y_train)
y_pred= RFC.predict(x_test_scaler)
p = RFC.score(x_test_scaler,y_test)
print("RandomForestClassifier:")
print(p)

print('Classification Report\n', classification_report(y_test, y_pred))
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))

cm = confusion_matrix(y_test, y_pred)
print(cm)


filename = 'model_RFC.pkl'
pickle.dump(RFC, open(filename, 'wb'))