import numpy
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics


weather = ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
temp = ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

#encode weather values and tempture values with numbers according to number of classes
le = preprocessing.LabelEncoder()

weatherEncoded = le.fit_transform(weather)
tempEncoded = le.fit_transform(temp)
Y = le.fit_transform(play)

#merge weather and tempture into pairs
X = list(zip(weatherEncoded,tempEncoded))

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.2)

#training the model
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

ypred = classifier.predict(X_test)
print("predection of the test-set : ",ypred)

res = metrics.accuracy_score(Y_test,ypred)
print("accuracy score : ",res)