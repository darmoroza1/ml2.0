from hmmlearn import hmm
from sklearn.metrics import accuracy_score
import numpy as np


def train_hmm(X_train, y_train, n_components=4):
    model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000, init_params="cmw")

    model.startprob_ = np.ones(n_components) / n_components
    model.transmat_ = np.ones((n_components, n_components)) / n_components

    lengths = [len(x) for x in X_train]
    X_train_concat = np.concatenate(X_train).reshape(-1, 1)

    model.fit(X_train_concat, lengths)
    return model


def evaluate_hmm(model, X_test, y_test):
    lengths = [len(x) for x in X_test]
    X_test_concat = np.concatenate(X_test).reshape(-1, 1)

    y_pred = model.predict(X_test_concat, lengths)
    y_pred_split = np.split(y_pred, np.cumsum(lengths)[:-1])

    y_pred_labels = [np.bincount(pred).argmax() for pred in y_pred_split]
    return accuracy_score(y_test, y_pred_labels)
