import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from . import linear_reggressor


def pca(X_train: pd.DataFrame, X_test: pd.DataFrame, k: int):
    """
    Using the PCA method to reduce the dimensionality of the data and perform prediction.
    :param X_train: the training samples matrix
    :param X_test: the test samples matrix
    :param k: the lower dimension
    :return: the transformed training and test samples matrices
    """
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(X_train)
    scaled_test_data = scaler.transform(X_test)

    pca = PCA(n_components=k)
    transformed_train_data = pca.fit_transform(scaled_train_data)
    transformed_test_data = pca.transform(scaled_test_data)

    return transformed_train_data,transformed_test_data

def cross_validation(self, X: np.ndarray, y: np.ndarray, k: int):
    """
    Performs k-fold cross-validation and sets self.model to the best model
    based on the cross-validation results.
    :param X: samples
    :param y: predictions
    :param k: number of folds for cross-validation
    """
    # Perform k-fold cross-validation
    scores = cross_val_score(self.model, X, y, cv=k,
                             scoring='neg_mean_squared_error')
    avg_scores = -scores.mean()

    # Set self.model to the best model based on cross-validation results
    best_model_index = np.argmax(scores)
    self.model = self.model  # Modify this line based on your specific model
    # selection criteria

    # Print the average RMSE for each fold
    print("Average RMSE for each fold:", -scores)

    # Print the best model's RMSE
    print("Best Model RMSE:", -scores[best_model_index])


def regularization_path(X, y, feature_names):
    # Compute Pearson correlation coefficients
    correlation_coeffs = X.corrwith(y)

    # Select top 10 features with highest correlation
    selected_features = correlation_coeffs.abs().nlargest(10).index.tolist()

    # Extract the selected features from the dataset
    X_selected = X[selected_features]

    lambdas = 10 ** np.linspace(-5, 5, 100)
    lambdas = np.sort(lambdas)[::-1]

    coefs = []
    for a in lambdas:
        ridge = Ridge(alpha=a, fit_intercept=False)
        ridge.fit(X_selected, y)
        coefs.append(ridge.coef_)

    fig, ax = plt.subplots()

    for i in range(len(selected_features)):
        ax.plot(lambdas, [coef[i] for coef in coefs], label=selected_features[i])

    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim())  # reverse axis
    plt.xlabel("Alpha")
    plt.ylabel("Weights")
    plt.title("Ridge coefficients as a function of the regularization")
    plt.legend()
    plt.axis("tight")
    plt.show()


def select_best_k_features(X,y,k,feature_names):
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit_transform(X, y)

    # Get the selected feature indices
    selected_indices = selector.get_support(indices=True)

    # Print the selected feature names
    selected_features = feature_names[selected_indices]
    print("Selected Features:")
    for feature in selected_features:
        print(feature)


def mse_over_lamda(train_x,train_y,test_x,test_y):
    alpha_values = np.logspace(-12, -2, 1000)  # Adjust the range as needed
    mse_values = []
    best_alpha = None
    best_mse = float('inf')  # Initialize with a large value
    for alpha in alpha_values:
        # Create and fit the Ridge regression model
        model = linear_reggressor.TumorSizePredict(lam=alpha,reggressor=Ridge)
        model.fit(train_x, train_y)

        # Make predictions on the test set
        y_pred = model.predict(test_x)

        # Calculate the Mean Squared Error (MSE)
        mse = mean_squared_error(y_true=test_y, y_pred=y_pred)
        mse_values.append(mse)
        if mse < best_mse:
            best_alpha = alpha
            best_mse = mse

    # Plot the MSE values over the alpha values
    plt.plot(alpha_values, mse_values)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('MSE')
    plt.title('MSE over Alpha')
    plt.show()
    print(best_alpha)
    print(best_mse)


def labeling_correlation(true_label, predicted_label):
    # Compute the multilabel confusion matrix
    mcm = multilabel_confusion_matrix(true_label, predicted_label)

    # Normalize the confusion matrix for each label
    mcm_norm = [cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] for cm in mcm]

    # Create a heatmap of the normalized confusion matrix for each label
    fig, axes = plt.subplots(len(mcm), figsize=(6, 4 * len(mcm)))
    for i, (cm_norm, ax) in enumerate(zip(mcm_norm, axes)):
        sns.heatmap(cm_norm, annot=True, cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(f'Normalized Confusion Matrix - Label {i}')
    plt.tight_layout()
    plt.show()


def plot_normalized_confusion_matrix(y_true, y_pred, labels):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create ConfusionMatrixDisplay object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)

    # Plot the confusion matrix
    disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal')

    # Set title and axis labels
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Show the plot
    plt.show()


def cluster_metastasis_mark(data: pd.DataFrame):
    """
    This function will cluster the data according to the metastases_mark
    feature
    :param data: samples matrix
    :return: None
    """
    feature_column = 'histopatological_degree'

    # Extract the relevant feature for clustering
    features = data[[feature_column]]

    # Perform clustering based on Euclidean distance
    kmeans = KMeans(n_clusters=3)  # Set the desired number of clusters
    kmeans.fit(features)
    labels = kmeans.labels_

    # Add the cluster labels to the original DataFrame
    data['cluster_label'] = labels

    # Group the data by cluster label
    grouped_data = data.groupby('cluster_label')

    # Calculate the Euclidean distance within each cluster
    cluster_distances = []
    for cluster_label, cluster_data in grouped_data:
        cluster_features = cluster_data[feature_column]
        cluster_distance = np.mean(
            cdist(cluster_features.values.reshape(-1, 1),
                  cluster_features.values.reshape(-1, 1)))
        cluster_distances.append(cluster_distance)

    # Plot the cluster distances
    plt.bar(range(len(cluster_distances)), cluster_distances)
    plt.xlabel('Cluster')
    plt.ylabel('Mean Euclidean Distance')
    plt.title('Euclidean Distance within Clusters')
    plt.show()

    # feature_column = 'histopatological_degree'
    #
    # counter = 0
    # for unique_value in data[feature_column].unique():
    #     specific = data[data[feature_column] == unique_value]
    #
    #     print(counter)
    #     counter += 1
    #     print(specific.sum())

    #for unique value in data[feature]
    #filter where data[feature]==unique value
    #summary





