import numpy as np
from sklearn.metrics import roc_auc_score


class D1ffic00ltLinearRegression(object):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.X_train = None
        self.y_test = None
        self.X_test = None
        self.X_predict = None
        self.w = None
        self.n = None
        self.k = None
        self.pred = None
        self.L = 1

    def fit(self, X, y):
        self.n, self.k = X.shape
        self.X_train = X
        if self.fit_intercept:
            self.X_train = np.hstack((np.ones((self.n, 1)), X))

        self.w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        return

    def fit_with_l2(self, X, y, L: float = 1, predict: bool = False):
        self.L = L
        self.n, self.k = X.shape
        self.X_train = X
        if self.fit_intercept:
            self.X_train = np.hstack((np.ones((self.n, 1)), X))
        self.w = np.linalg.inv(self.X_train.T @ self.X_train + self.L * np.eye(self.k + 1)) @ self.X_train.T @ y
        if predict:
            if self.X_test is None:
                raise AttributeError("self.X_test is None. Use set_x_test(X).")
            if self.y_test is None:
                raise AttributeError("self.y_test is None. Use set_y_true(y).")
            return roc_auc_score(self.y_test, self.predict(self.X_test))

    def predict(self, X):
        self.n, self.k = X.shape
        self.X_predict = X
        if self.fit_intercept:
            self.X_predict = np.hstack((np.ones((self.n, 1)), self.X_predict))
        self.pred = np.dot(self.X_predict, self.w)
        return self.pred

    def get_weights(self):
        return self.w

    def set_y_true(self, y):
        self.y_test = y

    def set_x_test(self, X):
        self.X_test = X