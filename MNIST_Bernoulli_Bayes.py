# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:42:54 2020

@author: scant
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import seaborn as sns
# import os

train_data = pd.read_csv("MNIST_train.csv")
test_data = pd.read_csv("MNIST_test.csv")
# print(train_data.head())
# print(test_data.head())

X_train = train_data.to_numpy()
X_test = test_data.to_numpy()
# print(X_train)
# print(X_test)

y_train = X_train[:, 0]  # Define y as first column of X
y_test = X_test[:, 0]  # Define y as first column of X
# print(y_train)
# print(y_test)

X_train = X_train[:, 1:]  # Chop off last column of X now that we've grabbed the labels
X_test = X_test[:, 1:]  # Chop off last column of X now that we've grabbed the labels
print(X_train)
print(len(X_train[0]))
print(X_test)
print(len(X_test[0]))

# Convert to 0 or 1 for Bernoulli dist.
Bern_cut = 100
X_train[np.where(X_train <= Bern_cut)] = 0
X_train[np.where(X_train > Bern_cut)] = 1
X_test[np.where(X_test <= Bern_cut)] = 0
X_test[np.where(X_test > Bern_cut)] = 1

for i in range(10):  # Show average for each numeral
    plt.figure()
    plt.imshow(np.median(X_train[y_train == i, :], axis=0).reshape(28, 28))
    plt.axis('off')


class BernoulliNB():

    def fit(self, X, y, epsilon=1e-10):
        self.likelihoods = dict()
        self.priors = dict()
        # Determine the set of unique catagories/labels/features
        self.K = set(y.astype(int))

        for k in self.K:  # Evaluate likelihoods/priors for each case
            X_k = X[y == k, :]
            p = len(X_k)/len(X)  # priors prob.
            # Do summation counting obvervations inside fit so it can be passed
            self.likelihoods[k] = (np.sum(X_k, axis=0) / len(X_k)) + epsilon
            self.priors[k] = p  # N_k/N

    def predict(self, X):  # Calculate probability matrix on new input
        N, D = X.shape
        P_hat = np.zeros((N, len(self.K)))

        for k, l in self.likelihoods.items():
            # Predict using Bernoulli prob. dens. func. on likelihood
            P_hat[:, k] = np.sum(X*np.log(l), 1) + np.sum((1-X)*np.log(1-l), 1) + np.log(self.priors[k])

        return P_hat.argmax(axis=1)  # return y_hat, i.e. argmax(P_hat)


def accuracy(y, y_hat):
    return np.mean(y == y_hat)


# Take training and test set and evaluate

bnb = BernoulliNB()
bnb.fit(X_train, y_train)  # Generate the probability matrix for training set X
y_hat = bnb.predict(X_test)  # Here X should be the test set

print("\nAccuracy: " + str(accuracy(y_test, y_hat)*100) + "%")

# Make Covarience Matrix
plt.figure(figsize=(9, 6))
y_true = pd.Series(y_test, name="Actual Label")
y_pred = pd.Series(y_hat, name="Predicted Label")
sns.heatmap(pd.crosstab(y_true, y_pred), annot=True, fmt="d", linewidths=0.25)
plt.ylim(len(set(y_test)), 0)  # Fix limits, matplotlib bugged (ver. 3.11)
