from pandas import read_csv

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.preprocessing import MinMaxScaler

import os







filname = "wisc_bc_data.csv"

dataset = read_csv(filname)



array = dataset.values



X_train = array[0:469, 2:32]

X_validation = array[469:570, 2:32]

Y_train = array[0:469, 1]

Y_validation = array[469:570, 1]





minmaxscale = MinMaxScaler().fit(X_train)

X_train = minmaxscale.transform(X_train)

X_validation = minmaxscale.transform(X_validation)



model = KNeighborsClassifier(n_neighbors=21)  #n_neighbors=1

kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

print('%s: %f (%f)' % ("kNN", cv_results.mean(), cv_results.std()))



model.fit(X_train, Y_train)

predictions = model.predict(X_validation)



print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))