import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, Ridge


class TumorSizePredict:
    def __init__(self,lam=0,reggressor=None):
        if reggressor is not None and lam != 0:
            self.regularaized=True
            self.model = reggressor(lam, max_iter=5000)
        else:
            self.model = LinearRegression()
            self.regularaized = False
        self.lamda_reg_param=lam
        self.regg_loss=np.inf

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        fits a linear regression model
        :param X: samples
        :param y: predictions
        """
        self.model.fit(X, y)

    def predict(self,X: np.ndarray)-> np.ndarray:
        """
        using linear regression to predict the labels
        :param X: samples matrix
        :return: predicted labels for the X matrix
        """
        return self.model.predict(X)

    def pca(self, X: np.ndarray, k: int):
        """
        using the PCA method to get a lower dimension to the data
        :param k: the lower dimension
        :param X: the samples matrix
        :return: the sample matrix in the lower dimension
        """
        # scaling the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X=X)

        # creating PCA object
        pca = PCA(n_components=k)

        return pca.fit_transform(scaled_data)




