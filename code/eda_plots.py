import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster import hierarchy
from sklearn.linear_model import RidgeCV


def kmeans_cluster(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X)
    return labels


def spectral_cluster(X, n_clusters=3):
    spectral = SpectralClustering(n_clusters=n_clusters)
    labels = spectral.fit_predict(X)
    return labels

"""
def eda_plot_cancer_patient(X, y):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    fig = go.Figure(data=go.Scatter3d(
        x=df_pca['PC1'],
        y=df_pca['PC2'],
        z=df_pca['PC3'],
        mode='markers',
        marker=dict(
            color=y.apply(lambda x: 'red' if x > 0 else 'green'),
            size=2,
            opacity=0.8
        )
    ))

    # Update plot aesthetics
    fig.update_layout(
        title='PCA 3D Plot',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        )
    )

    # Show the plot
    fig.show()
"""


def cluster_samples(data):
    # Calculate the Pearson correlation matrix
    corr_matrix = data.corr(method='pearson')
    print(corr_matrix)
    # Calculate the pairwise distances based on the correlation matrix
    dist_matrix = 1 - np.abs(corr_matrix)

    # Perform hierarchical clustering using the pairwise distances
    linkage = hierarchy.linkage(dist_matrix, method='ward')
    clusters = hierarchy.fcluster(linkage, t=0.5, criterion='distance')

    # Visualize the clustering result
    sns.clustermap(data, method='ward', metric='correlation', cmap='coolwarm',
                   row_cluster=True, col_cluster=False, figsize=(10, 8))

    # Set the title of the plot
    plt.title("Hierarchical Clustering of Samples")

    # Show the plot
    plt.show()

    return clusters


def eda_plot_cancer_patient(X, y):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])

    # Perform K-means clustering
    kmeans_labels = kmeans_cluster(X_pca)
    print(kmeans_labels)

    # Define colors for clusters
    cluster_colors = ['red', 'green', 'blue']  # Add more colors if needed

    fig = go.Figure()

    # Add scatter trace for each cluster in K-means
    for cluster in range(max(kmeans_labels) + 1):
        cluster_indices = kmeans_labels == cluster
        fig.add_trace(go.Scatter3d(
            x=df_pca.loc[cluster_indices, 'PC1'],
            y=df_pca.loc[cluster_indices, 'PC2'],
            z=df_pca.loc[cluster_indices, 'PC3'],
            mode='markers',
            marker=dict(
                color=cluster_colors[cluster],
                size=2,
                opacity=0.8
            ),
            name=f'K-means Cluster {cluster}'
        ))

    # Add scatter trace for each cluster in Spectral clustering
    """for cluster in range(max(spectral_labels) + 1):
        cluster_indices = spectral_labels == cluster
        fig.add_trace(go.Scatter3d(
            x=df_pca.loc[cluster_indices, 'PC1'],
            y=df_pca.loc[cluster_indices, 'PC2'],
            z=df_pca.loc[cluster_indices, 'PC3'],
            mode='markers',
            marker=dict(
                color=cluster_colors[cluster],
                size=2,
                opacity=0.8
            ),
            name=f'Spectral Cluster {cluster}'
        ))"""

    # Update plot aesthetics
    fig.update_layout(
        title='Clustering Results - PCA 3D Plot',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        )
    )

    # Show the plot
    fig.show()

def pearson_values_correlation(X,y):
    corr_matrix = X.corrwith(y)

    # Create a correlation dataframe
    corr_df = pd.DataFrame(corr_matrix, columns=['correlation'])

    # Create a heatmap of the correlation values
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Pearson Correlation Heatmap with tummor size')
    plt.show()

def eda_plot_cancer_patient_spec(X, y):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])

    # Perform Spectral clustering
    spectral_labels = spectral_cluster(X_pca)
    print(spectral_labels)

    # Define colors for clusters
    cluster_colors = ['red', 'green', 'blue']  # Add more colors if needed

    fig = go.Figure()

    # Add scatter trace for each cluster in Spectral clustering
    for cluster in range(max(spectral_labels) + 1):
        cluster_indices = spectral_labels == cluster
        fig.add_trace(go.Scatter3d(
            x=df_pca.loc[cluster_indices, 'PC1'],
            y=df_pca.loc[cluster_indices, 'PC2'],
            z=df_pca.loc[cluster_indices, 'PC3'],
            mode='markers',
            marker=dict(
                color=cluster_colors[cluster],
                size=2,
                opacity=0.8
            ),
            name=f'Spectral Cluster {cluster}'
        ))

    # Update plot aesthetics
    fig.update_layout(
        title='Clustering Results - PCA 3D Plot',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        )
    )

    # Show the plot
    fig.show()


def pear_ridg_cor(X,y):
    correlation_coeffs = X.corrwith(y)

    # Select feature(s) with highest correlation
    selected_features = [correlation_coeffs.idxmax()]

    # Perform ridge regression with cross-validation
    ridge = RidgeCV(alphas=np.logspace(-5, 5, 100))
    ridge.fit(X[selected_features], y)

    # Get regularization path and coefficient values
    alphas = ridge.alphas_
    coef_path = ridge.coef_

    # Plot the regularization path
    plt.figure(figsize=(8, 6))
    plt.plot(np.log10(alphas), coef_path.T)
    plt.xlabel('log(alpha)')
    plt.ylabel('Coefficient Value')
    plt.title('Ridge Regularization Path for Tumor Size')
    plt.legend(selected_features)
    plt.grid(True)
    plt.show()


def summarize_dataframe(df):
    summary = {}

    # Perform PCA on numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    pca = PCA(n_components=2)
    X_pca_2d = pca.fit_transform(numeric_cols)

    # Perform K-means clustering on PCA 2D
    kmeans = KMeans(n_clusters=2)
    cluster_labels_2d = kmeans.fit_predict(X_pca_2d)

    # Plot PCA 2D with cluster labels
    fig_2d = px.scatter(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1],
                        color=cluster_labels_2d)
    fig_2d.update_layout(title='PCA 2D with Clustering')
    fig_2d.show()

    # Perform PCA on numeric columns for 3D
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(numeric_cols)

    # Perform K-means clustering on PCA 3D
    kmeans_3d = KMeans(n_clusters=2)
    cluster_labels_3d = kmeans_3d.fit_predict(X_pca_3d)

    # Plot PCA 3D with cluster labels
    fig_3d = px.scatter_3d(
        x=X_pca_3d[:, 0],
        y=X_pca_3d[:, 1],
        z=X_pca_3d[:, 2],
        color=cluster_labels_3d
    )
    fig_3d.update_layout(title='PCA 3D with Clustering')
    fig_3d.show()

    return summary


def cluster_data(train_x, train_y, feature_names):
    # Step 3: Apply dimensionality reduction using PCA
    pca = PCA(n_components=3)  # Choose the desired number of components
    X_pca = pca.fit_transform(train_x)

    # Step 4: Apply clustering algorithm (Agglomerative Clustering in this example)
    n_clusters = 2
    hier = AgglomerativeClustering(n_clusters=n_clusters, affinity="euclidean", linkage="ward")
    cluster_labels = hier.fit_predict(train_x)

    # Step 5: Evaluate and Interpret (if necessary)
    silhouette_avg = silhouette_score(train_x, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg}")

    # Visualize the data points with cluster labels in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=cluster_labels, cmap='viridis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('Clustering Results - PCA 3D Scatter Plot')
    plt.colorbar(scatter)
    plt.show()


def plot_correlation_heatmap(data):
    # Calculate the Pearson correlation matrix
    corr_matrix = data.corr(method='pearson')

    # Generate a mask for the upper triangle (to avoid duplicate values)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Set up the matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Generate the heatmap using seaborn
    sns.heatmap(corr_matrix, mask=mask, annot=False, fmt=".2f",
                cmap='coolwarm',
                cbar_kws={'shrink': 0.8}, square=True)

    # Set the title of the plot
    plt.title("Pearson Correlation Heatmap")

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.show()


def cluster_similar_metastasis(data: pd.DataFrame):
    """
    This function will cluster samples with similar metastases
    :param data: the sample matrix
    :return:
    """



