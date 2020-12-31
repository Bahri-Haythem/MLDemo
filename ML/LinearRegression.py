import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style

#read csv file with pandas
data = pd.read_csv("LinearRegression.csv",sep=";")

data = data[["G1","G2","G3","traveltime","failures","health"]]
predict = "G3"

#dataset labels or the independent variables
x = np.array(data.drop([predict],1))
#label to predict or the dependent variable
y = np.array(data[predict])

#splitting the data into train_set and test_set
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
linear = linear_model.LinearRegression()
#training the model
linear.fit(x_train,y_train)
#accuracy score
acc = linear.score(x_test,y_test)

"""
#printing the test_set prediction and their true value
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])
"""

style.use("ggplot")
p="failures"
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("final grade")
pyplot.show()
