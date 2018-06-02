#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing


def load_train_test_data(train_ratio=.5):
    data = pandas.read_csv('./ENB2012_data.csv')
   
    feature_col = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    label_col = ['Y1']
    X = data[feature_col]   
    y = data[label_col]
    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale


def gradient_descent(X, y, alpha = .001, iters = 100000, eps=1e-4):
    # TODO: fill this procedure as an exercise
    n, d = X.shape ##(384, 9)
    theta = numpy.matrix(numpy.zeros((d , 1))) ##(9, 1)
    oneX = numpy.matrix(X)
    l = 1 #lambda   
    for iter in range(iters):
        g = numpy.transpose(oneX) * (oneX*theta - y) + l * theta 
        diff = alpha * g
        theta = theta - diff
        k = max(abs(diff[j]) for j in range(d))
        if k < eps:
            return theta   
    return theta


def predict(X, theta):
    return numpy.dot(X, theta)


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)
    X_train_scale = numpy.concatenate((numpy.ones((len(X_train_scale),1)), X_train_scale), axis = 1)
    X_test_scale = numpy.concatenate((numpy.ones((len(X_test_scale),1)), X_test_scale), axis = 1)
    theta = gradient_descent(X_train_scale, y_train)
    y_hat = predict(X_train_scale, theta)
    print("Linear train R^2: %f" % (sklearn.metrics.r2_score(y_train, y_hat)))
    y_hat = predict(X_test_scale, theta)
    print("Linear test R^2: %f" % (sklearn.metrics.r2_score(y_test, y_hat)))


if __name__ == "__main__":
    main(sys.argv)


