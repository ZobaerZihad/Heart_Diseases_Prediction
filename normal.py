import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
import pickle

heart_data = pd.read_csv("heart_.csv")

print(heart_data.head())
print(heart_data.info())


heart_data['sex'][heart_data['sex'] == 0] ='female'
heart_data['sex'][heart_data['sex'] == 1] ='male'

heart_data['cp'][heart_data['cp'] == 0] = 'typical angina'
heart_data['cp'][heart_data['cp'] == 1] = 'atypical angina'
heart_data['cp'][heart_data['cp'] == 2] = 'non-anginal'
heart_data['cp'][heart_data['cp'] == 3] = 'asymptotic'

heart_data['fbs'][heart_data['fbs'] == 0] = 'lower than 120ml/mg'
heart_data['fbs'][heart_data['fbs'] == 1] = 'upper than 120ml/mg'

heart_data['exang'][heart_data['exang'] == 0] = 'no'
heart_data['exang'][heart_data['exang'] == 1] = 'yes'

heart_data['slope'][heart_data['slope'] == 1] = 'upsloping'
heart_data['slope'][heart_data['slope'] == 2] = 'flat'
heart_data['slope'][heart_data['slope'] == 3] = 'downsloping'

heart_data['thal'][heart_data['thal'] == 1] = 'normal'
heart_data['thal'][heart_data['thal'] == 2] = 'fixed defect'
heart_data['thal'][heart_data['thal'] == 3] = 'reversable defect '

print(heart_data.dtypes)
print(heart_data.head())

ObjToCat = ['sex','cp','fbs','exang','slope','thal']

for i in ObjToCat:
    heart_data[i]=heart_data[i].astype('category')


print(heart_data.info())

#Creating Dummy

heart_data = pd.get_dummies(heart_data,drop_first=True)
print(heart_data.head())

x=heart_data['chol']
y=heart_data['target']
#plt.scatter(x,y)
#plt.show()


X=heart_data.drop('target',axis=1)
Y=heart_data['target']


print(X.head())


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier()


model.fit(X_train,Y_train)

print(model.score(X_test,Y_test)*100)


y_predicted = model.predict(X_test)

cm=confusion_matrix(Y_test,y_predicted)
sns.heatmap(cm,annot=True,fmt='d',cbar=False,annot_kws={'size':12})
plt.show()
print(classification_report(Y_test,y_predicted))

with open('model_pickle', 'rb') as f:
    model = pickle.load(f)
prediction = model.predict(X_test)

print("Model Test")
print(prediction)