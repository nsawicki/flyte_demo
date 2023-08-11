import os
import json
from typing import Tuple

import numpy as np
import pandas as pd
import sklearn

from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

import flytekit.extras.sklearn
from flytekit import task, workflow


@task
def get_data() -> Tuple[np.ndarray,np.ndarray]:
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    return(digits.images.reshape((n_samples,-1)),digits.target)

@task
def get_train_test(input_digits: np.ndarray,labels:np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
                            input_digits, labels, test_size=0.5, shuffle=False
                                            )
    return(X_train,X_test,y_train,y_test)

@task
def get_classifier() -> svm._classes.SVC:
    return svm.SVC(gamma=.001)

@task
def train_test(X_train:np.ndarray, y_train:np.ndarray,X_test:np.ndarray,clf:svm._classes.SVC) -> np.ndarray:
    clf.fit(X_train,y_train)
    return clf.predict(X_test)

@task
def report_accuracy(y_pred:np.ndarray,y_test:np.ndarray):
    print('Accuracy: ' + str(len(y_test) - sum(abs(y_pred-y_test))))
    print('Out of ' + str(len(y_test)))

@workflow
def full_pipeline():

    dataset,labels = get_data()
   
    (X_train, X_test, y_train, y_test) = get_train_test(input_digits=dataset,labels=labels)
    
    clf = get_classifier()
    
    predictions = train_test(X_train=X_train,y_train=y_train,X_test=X_test,clf=clf)

    report_accuracy(y_pred=predictions,y_test=y_test)

if __name__ == '__main__':

    digits = datasets.load_digits()
    print(type(digits.target))
    n_samples = len(digits.images)
    print(type(digits))
    print(type(digits.images.reshape((n_samples,-1))))

    data = digits.images.reshape((n_samples,-1))

    X_train, X_test, y_train, y_test = train_test_split(
                data, digits.target, test_size=0.5, shuffle=False
                )

    print(type(X_train))
    print(type(y_test))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    print(type(clf))
