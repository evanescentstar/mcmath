## module containing functions for / or data science analysis tools

import numpy as np
# import numpy.ma
import scipy as sp
import scipy.signal as sig
import scipy.stats as stats
import datetime
from datetime import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
from mcmath import rs1

def logress1(X, y, multi_class='auto', solver='lbfgs'):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, balanced_accuracy_score, recall_score

    if np.unique(y).size > 2:
        ### using LabelEncoder()
        lab1 = LabelEncoder()
        y4 = lab1.fit_transform(y)
        y = y4.copy()

    X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rs1())

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train0)
    X_test = sc.transform(X_test0)

    classifier = LogisticRegression(multi_class=multi_class, solver=solver, random_state=rs1())
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    yp0 = classifier.predict(X_train)

    return (X_train, X_test, y_train, y_test, sc, classifier, y_pred, yp0)


def dectreeclass1(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, balanced_accuracy_score, recall_score

    X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rs1())

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train0)
    X_test = sc.transform(X_test0)

    classifier = DecisionTreeClassifier(criterion='entropy', random_state=rs1())
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    yp0 = classifier.predict(X_train)

    return (X_train, X_test, y_train, y_test, sc, classifier, y_pred, yp0)

def svmclass1(X, y, kernel='rbf', prob=False, dfs='ovr'):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from sklearn.svm import SVC

    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, balanced_accuracy_score, recall_score

    X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rs1())

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train0)
    X_test = sc.transform(X_test0)

    classifier = SVC(kernel = kernel, probability=prob, decision_function_shape=dfs, random_state = rs1())
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    yp0 = classifier.predict(X_train)

    scores(y_test,y_pred)

    return (X_train, X_test, y_train, y_test, sc, classifier, y_pred, yp0)

def scores(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, balanced_accuracy_score, recall_score
    print('accuracy_score: %.3f' % accuracy_score(y_test, y_pred))
    print('balanced_accuracy_score: %.3f' % balanced_accuracy_score(y_test, y_pred))
    if np.unique(y_test).size > 2:
        print('precision_score: %.3f' % precision_score(y_test, y_pred, average='weighted'))
    else:
        print('precision_score: %.3f' % precision_score(y_test, y_pred))
    if np.unique(y_test).size > 2:
        print('recall_score: %.3f' % recall_score(y_test, y_pred, average='weighted'))
    else:
        print('recall_score: %.3f' % recall_score(y_test, y_pred))
    return None
