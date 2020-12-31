import pandas as pd
import numpy as np
from sklearn import preprocessing,neighbors
from sklearn import model_selection

#read csv file with pandas
data = pd.read_csv('KNN.data',sep=',')

#encode values with numbers according to number of classes
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
class_ = le.fit_transform(list(data["class"]))
persons = le.fit_transform(list(data["persons"]))
door = le.fit_transform(list(data["door"]))

predict ="class"

#merge labels into tuples
x = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(class_)

#splitting the data into train_set and test_set
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)

#creating a vector to save the accuracies and choosing the best one with the for loop
ks = 20
meanAcc = np.zeros((ks-1))

for n in range(1,9):
    #creating the model with number of neighbors increasing each iteration
    model = neighbors.KNeighborsClassifier(n_neighbors=n)
    #training the model
    model.fit(x_train,y_train)
    # accuracy score
    acc = model.score(x_test,y_test)
    print(n , " and " , acc)
    #adding accuracy to the table meanAcc
    meanAcc[n-1] = acc

#best accuracy is the max value of meanAcc
print("the best acc is ",meanAcc.max()," and the k is ",meanAcc.argmax()+1)
