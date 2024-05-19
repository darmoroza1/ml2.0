import numpy as np
from scipy.fft import fft
from sklearn.metrics import accuracy_score

def fourier_transform(signal):
    return np.abs(fft(signal))


def evaluate_fourier_hmm(model, X_test, y_test):
    lengths = [len(x) for x in X_test]
    X_test_concat = np.concatenate(X_test).reshape(-1, 1)

    y_pred = model.predict(X_test_concat, lengths)
    y_pred_split = np.split(y_pred, np.cumsum(lengths)[:-1])

    y_pred_labels = [np.bincount(pred).argmax() for pred in y_pred_split]
    return accuracy_score(y_test, y_pred_labels)
