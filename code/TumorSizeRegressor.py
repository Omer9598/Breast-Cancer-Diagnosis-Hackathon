import xgboost as xgb
from sklearn.linear_model import Ridge


class TumorSizeRegressor:
    def __init__(self, lam=0.01, n_estimators=50, max_depth=4):
        self.have_cancer_model = xgb.XGBClassifier(n_estimators=n_estimators,
                                                   max_depth=max_depth)
        self.cancer_size_model = Ridge(alpha=lam)

    def fit(self, X, y):
        y_binary = y.apply(lambda x: 1 if x > 0 else 0)
        self.have_cancer_model.fit(X, y_binary)

        y_with_cancer_size = y[y > 0]
        X_with_cancer_size = X.loc[y_with_cancer_size.index]

        self.cancer_size_model.fit(X_with_cancer_size, y_with_cancer_size)

    def predict(self, X):
        cancer_size_predict = self.cancer_size_model.predict(X)
        have_cancer_predict = self.have_cancer_model.predict(X)

        return cancer_size_predict * have_cancer_predict

    def loss(self, X, y):
        y_predicted = self.predict(X)
        return ((y_predicted - y) ** 2).mean()
