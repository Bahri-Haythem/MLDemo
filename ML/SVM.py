from sklearn import datasets
from sklearn import svm
import sklearn

cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target

#splitting the data into train_set and test_set
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
clf = svm.SVC()
#training the model
clf.fit(x_train,y_train)

#predicting the test_set
y_pred = clf.predict(x_test)
print(y_pred)
#accuracy score
acc = sklearn.metrics.accuracy_score(y_test,y_pred)
print(acc)