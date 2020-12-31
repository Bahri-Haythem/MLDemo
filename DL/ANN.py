import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer

# Importing the dataset
dataset = pd.read_csv('ANN.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')

labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])

X = np.array(columnTransformer.fit_transform(X), dtype = np.int)
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6, activation="relu", input_dim=11, kernel_initializer="uniform"))

classifier.add(Dense(6, activation="relu", kernel_initializer="uniform"))

classifier.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

classifier.compile(optimizer = "adam",loss = "binary_crossentropy",metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=10,epochs=20)

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
