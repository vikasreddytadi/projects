import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import plotly.express as px



df = pd.read_excel('C:/Users/Aditya/Downloads/New folder/wine quality/winequality-red.xlsx')



# See the number of rows and columns
print("Rows, columns: " + str(df.shape))



# See the first five rows of the dataset
df.head()



# Missing Values
print(df.isna().sum())
fig = px.histogram(df,x='quality')
fig.show()



# Create Classification version of target variable
df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]


# Separate feature variables and target variable
X = df.drop(['quality','goodquality'], axis = 1)
y = df['goodquality']


# See proportion of good vs bad wines
df['goodquality'].value_counts()


# Normalize feature variables
from sklearn.preprocessing import StandardScaler
X_features = X
X = StandardScaler().fit_transform(X)


# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=7)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print(classification_report(y_test, y_pred2))
