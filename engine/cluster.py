import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class ClusterModule:

    desc = """
Unsupervised clustering.

Machine learning technique that groups similar data points together without using labeled data. It
identifies patterns and structures within the data to create clusters based on the intrinsic
properties and relationships among the data points."""

    models = {
        "DBSCAN": {
            "model": DBSCAN,
            "docstr": """
DBSCAN - Density-Based Spatial Clustering of Applications with Noise. Finds core samples of high
density and expands clusters from them. Good for data which contains clusters
of similar density.""",
            "kwargs": {
                "eps": """
eps: float, default=0.5

The maximum distance between two samples for one to be considered as in the neighborhood of the
other. This is not a maximum bound on the distances of points within a cluster. This is the most 
important DBSCAN parameter to choose appropriately for your data set and distance function.""",
                "min_sampes": """
min_samples: int, default=5

The number of samples (or total weight) in a neighborhood for a point to be considered as a core 
point. This includes the point itself. If min_samples is set to a higher value, DBSCAN will find 
denser clusters, whereas if it is set to a lower value, the found clusters will be more sparse.""",
                "metric": """
metric: str, default='euclidean'

The metric to use when calculating distance between instances in a feature array. It must be one of 
the 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'.""",
            },
        },
        "OPTICS": {
            "model": OPTICS,
            "docstr": """
OPTICS (Ordering Points To Identify the Clustering Structure), closely related to DBSCAN, finds core
sample of high density and expands clusters from them [1]. Unlike DBSCAN, keeps cluster hierarchy
for a variable neighborhood radius. Better suited for usage on large datasets than the current
sklearn implementation of DBSCAN.

[1] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel, and Jörg Sander. “OPTICS: ordering
points to identify the clustering structure.” ACM SIGMOD Record 28, no. 2 (1999): 49-60.""",
            "kwargs": {
                "min_samples": """
min_samples: int > 1 or float between 0 and 1, default=5

The number of samples in a neighborhood for a point to be considered as a core point. Also, up and
down steep regions can't have more than min_samples consecutive non-steep points. Expressed as an
absolute number or a fraction of the number of samples (rounded to be at least 2).""",
                "max_eps": """
max_eps: float, default=np.inf

The maximum distance between two samples for one to be considered as in the neighborhood of the
other. Default value of np.inf will identify clusters across all scales; reducing max_eps will
result in shorter run times.""",
                "metric": """
metric: str, default='minkowski'

Metric to use for distance computation. It must be one of  the 'cityblock', 'cosine', 'euclidean',
'l1', 'l2', 'manhattan'.""",
            },
        },
        "Agglomerative": {
            "model": AgglomerativeClustering,
            "docstr": """
Agglomerative Clustering recursively merges pair of clusters of sample data; uses linkage distance.""",
            "kwargs": {
                "n_clusters": """
n_clusters: int, default=2

The number of clusters to find.""",
                "metric": """
metric: str, default="euclidean"

Metric used to compute the linkage. Can be 'euclidean', 'l1', 'l2', 'manhattan', 'cosine'.""",
            },
        },
        "KMeans": {
            "model": KMeans,
            "docstr": """
The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance,
minimizing a criterion known as the inertia or within-cluster sum-of-squares (see below). This
algorithm requires the number of clusters to be specified. It scales well to large numbers of
samples and has been used across a large range of application areas in many different fields.""",
            "kwargs": {
                "n_clusters": """
n_clusters: int, default=8

The number of clusters to form as well as the number of centroids to generate.""",
            },
        },
    }

    scores = {
        "Davies-Bouldin": {
            "score": davies_bouldin_score,
            "docstr": """
The Davies-Bouldin score is defined as the average similarity measure of each cluster with its most
similar cluster, where similarity is the ratio of within-cluster distances to between-cluster
distances. Thus, clusters which are farther apart and less dispersed will result in a better score.

The minimum score is zero, with lower values indicating better clustering.""",
        },
        "Silhouette": {
            "score": silhouette_score,
            "docstr": """
The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean
nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) /
max(a, b). To clarify, b is the distance between a sample and the nearest cluster that the sample is
not a part of. Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels
<= n_samples - 1. 

The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative
values generally indicate that a sample has been assigned to the wrong cluster, as a different
cluster is more similar.""",
        },
        "Calinski-Harabasz": {
            "score": calinski_harabasz_score,
            "docstr": """
The Calinski-Harabasz index is the ratio of the sum of between-clusters dispersion and of
within-cluster dispersion for all clusters (where dispersion is defined as the sum of distances
squared).

The score is higher when clusters are dense and well separated, which relates to a standard concept
of a cluster.""",
        },
    }

    @classmethod
    def hopkins(self, X: pd.DataFrame, metric: str = "minkowski", samples: int = 30) -> float:
        """Computes Hopkins statistics H value to estimate cluster tendency of data set `X`.

        It acts as a statistical hypothesis test where the null hypothesis is that the data is
        generated by a Poisson point process and are thus uniformly randomly distributed. Under the
        null hypothesis of spatial randomness, this statistic has a Beta(m,m) distribution and will
        always lie between 0 and 1. The interpretation of H follows these guidelines:

        - Low values of H indicate repulsion of the events in X away from each other.
        - Values of H near 0.5 indicate spatial randomness of the events in X.
        - High values of H indicate possible clustering of the events in X. Values of H>0.75
          indicate a clustering tendency at the 90% confidence level

        We calculate Hopkins statistic `samples` times and then calculate the mean value of Hopkins
        statistics.

        For details see: https://journal.r-project.org/articles/RJ-2022-055/

        Args:
            X: Data set. Shape (n_samples, n_features).
            metric: Metric used to compute nearest neigbors.
            samples: Number of samples used to calculate the mean of Hopkins statistics.

        Returns:
            Hopkins statistics value.
        """
        assert isinstance(X, pd.DataFrame), "Expected `X` to be a DataFrame"

        from random import sample

        n, d = X.shape
        m = int(0.1 * n) if int(0.1 * n) > 0 else 1

        def hopkins_sample():
            # Generate a random sample X_tilde of m<<n data points sampled without replacement from X
            X_tilde = X.iloc[sample(range(n), m), :]

            # Generate a set Y of m uniformly randomly distributed points
            Y = np.random.uniform(low=X.min(axis=0).values, high=X.max(axis=0).values, size=(m, d))

            knn = NearestNeighbors(metric=metric).fit(X.values)
            # Define two distance measures,
            # ui - the minimum distance of yi ∈ Y to its nearest neighbor in X \in
            # wi - the minimum distance of xi_tilde ∈ X_tilde to its nearest neighbor xj ∈ X, xj != xi_tilde
            ui, _ = knn.kneighbors(Y, n_neighbors=1, return_distance=True)
            wi, _ = knn.kneighbors(X_tilde.values, n_neighbors=2, return_distance=True)
            wi = wi[:, 1]

            # FIXME: This is not numerically stable
            return 1 / (1 + (wi**d).sum() / (ui**d).sum())

        return np.mean([hopkins_sample() for _ in range(samples)])

    @classmethod
    def cluster(self, X: pd.DataFrame, method: str = "DBSCAN", **kwargs) -> np.ndarray[int]:
        """Performs clustering of unlabeled data stored in `X` using model passed as a `method`
        string along with appropriate kwargs. The method should be one of the keys of
        `ClusterModule.models` field. Available kwargs for each model are the keys of
        `ClusterModule.models[method]['kwargs']` field.

        Args:
            X: Data set. Shape (n_samples, n_features).
            method: Name of model used to perform clustering. Should be one of the keys of
                    `ClusterModule.models` field.

        Returns:
            Array of cluster labels (numbers) for every data point in X. Shape (n_samples,).
        """
        assert isinstance(X, pd.DataFrame), "Expected `X` to be a DataFrame"
        assert method in self.models, f"Unrecognized method. Should be one of {self.models.keys()}"
        assert all(arg in self.models[method]["kwargs"] for arg in kwargs), f"Unrecognized argument"

        model = self.models[method]["model"]
        model = model(**kwargs).fit(X)

        return model.labels_

    @classmethod
    def evaluate(self, X: pd.DataFrame, labels: np.ndarray[int], method: str = "Davies-Bouldin") -> float:
        """Computes an appropriate score specified by `method` argument to evaluate the performance
        of a clustering algorithm.

        Args:
            X: Data set. Shape (n_samples, n_features).
            labels: Array of cluster labels (numbers) for every data point in X. Shape (n_samples,).
            method: Name of score to return. Should be one of the keys of `ClusterModule.scores`
                    field.

        Returns:
            Score value.
        """
        assert isinstance(X, pd.DataFrame), "Expected `X` to be a DataFrame"
        assert isinstance(labels, np.ndarray), "Expected `labels` to be a np.ndarray[int]"
        assert method in self.scores, f"Unrecognized method. Should be one of {self.scores.keys()}"
        return self.scores[method]["score"](X, labels.astype(int))

    @classmethod
    def describe(self, X: pd.DataFrame, labels: np.ndarray[int]) -> dict[str, dict[str, str | pd.DataFrame]]:
        """Computes descriptive statistics (count, mean, std, min, max, q1, q2, q3) for every
        cluster of points.

        Args:
            X: Data set. Shape (n_samples, n_features).
            labels: Array of cluster labels (numbers) for every data point in X. Shape (n_samples,).

        Returns:
            Dict of key, value pairs where key is a label (int) of cluster and value is a dict with
            two keys: `"desc"` containing a short description of cluster and `"stat"` containing a
            pandas DataFrame with values of descriptive statistics for every column of `X`.
        """
        assert isinstance(X, pd.DataFrame), "Expected `X` to be a DataFrame"
        assert isinstance(labels, np.ndarray), "Expected `labels` to be a np.ndarray[int]"

        X = pd.concat([X, pd.DataFrame(labels.reshape(-1, 1), columns=["Label"])], axis=1)
        return {
            label: {
                "desc": "Noise" if label == -1 else f"Cluster {label}",
                "stat": X.loc[X["Label"] == label, X.columns != "Label"].describe(),
            }
            for label in np.unique(labels)
        }

    @classmethod
    def visualize(self, X: pd.DataFrame, labels: np.ndarray[int]):
        """TODO:...

        Args:
            X: Data set. Shape (n_samples, n_features).
            labels: Array of cluster labels (numbers) for every data point in X. Shape (n_samples,).

        Returns:
            Figure object.
        """
        pca = PCA(n_components=2, random_state=0).fit(X)
        X_pca = pca.transform(X)

        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        ax.scatter(x=X_pca.iloc[:, 0], y=X_pca.iloc[:, 1], c=labels)
        ax.set_xlabel("PCA comp. 0")
        ax.set_ylabel("PCA comp. 1")

        return fig
