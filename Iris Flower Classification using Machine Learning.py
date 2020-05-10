# %%
"""
## Import Dependencies 
"""

# %%
import pandas as pd
import numpy as np
import plotly
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as pyo
import cufflinks as cf  #join a pandas and plotpy 
from plotly.offline import init_notebook_mode, plot, iplot
from pandas.plotting import scatter_matrix


# %%
#use to connet a plotly in jupyter notebook and use in offline mode
pyo.init_notebook_mode(connected=True)
cf.go_offline()

# %%
"""
### Get Data form CSV
"""

# %%
iris = pd.read_csv('Iris.csv')
iris.head()

# %%
iris = iris.drop('Id', axis=1)

# %%
iris.head()

# %%
iris.shape

# %%
"""
## Visualization Data
"""

# %%
px.scatter(iris, x='Species', y='PetalWidthCm')

# %%
px.line(iris, x='Species', y='PetalWidthCm')

# %%
#scatter_matrix(iris, figsize=(12,8))

# %%
iris = iris.rename(columns={'SepalLengthCm':'SepalLength','SepalWidthCm':'SepalWidth',
                           'PetalLengthCm':'PetalLength','PetalWidthCm':'PetalWidth'})
iris.head()

# %%
"""
### Correlation Visualization
"""

# %%
px.scatter_matrix(iris, color='Species', title='Iris', dimensions=['SepalLength','SepalWidth',
                                                                  'PetalLength','PetalWidth'])

# %%
"""
### Change Feature and Labels
"""

# %%
x= iris.drop('Species', axis=1)

# %%
y = iris['Species'].copy()

# %%
"""
### Change a Label in Numerical Format
"""

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

# %%
y

# %%
"""
## Dived a data in test and train
"""

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

# %%
x_train.shape

# %%
y_train.size

# %%
"""
## Apply a DecisionTreeClassifier Model
"""

# %%
from sklearn import tree

dt_model = tree.DecisionTreeClassifier()
dt_model.fit(x_train, y_train)

# %%
from sklearn.metrics import accuracy_score

prediction_dt = dt_model.predict(x_test)
accuracy_dt = accuracy_score(y_test, prediction_dt)*100

# %%
accuracy_dt

# %%
"""
### Cross Validation
"""

# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt_model,x_train, y_train, scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores

# %%
y_test

# %%
prediction_dt

# %%
"""
## Apply a KNeighborsClassifier Model
"""

# %%
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(x_train, y_train)

# %%
prediction_knn = knn_model.predict(x_test)

# %%
accuracy_knn = accuracy_score(y_test, prediction_knn)*100

# %%
accuracy_knn

# %%
"""
### Cross Validation
"""

# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn_model,x_train, y_train, scoring='neg_mean_squared_error', cv=20)
rmse_scores = np.sqrt(-scores)
rmse_scores

# %%
y_test

# %%
prediction_knn

# %%
"""
## Apply KMeans Model
"""

# %%
from sklearn.cluster import KMeans

km_model = KMeans(n_clusters=3, random_state=2, n_jobs=3)
km_model.fit(x)

# %%
centers = km_model.cluster_centers_
centers

# %%
km_model.labels_

# %%
"""
## Creating Catagory
"""

# %%
catagory = ['Iris-Satosa','Iris-Versicolor','Iris-Virginica']

# %%
"""
### Test a random data using DecisionTreeClassifier Model
"""

# %%
data = 5.7,3,4.2,1.1

# %%
data_array = np.array([data])
data_array

# %%
predic = dt_model.predict(data_array)

# %%
print(catagory[int(predic[0])])

# %%
"""
### Test a random data using KNeighborsClassifier Model
"""

# %%
predic = knn_model.predict(data_array)

# %%
print(catagory[int(predic[0])])

# %%
"""
### Test a random data using KMeans Model
"""

# %%
predic = km_model.predict(data_array)

# %%
print(catagory[int(predic[0])])

# %%
