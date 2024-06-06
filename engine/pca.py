import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from matplotlib.figure import Figure


class PCAModule:

    desc = """
Principal component analysis (PCA).

Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a
lower dimensional space. The input data is centered but not scaled for each feature before applying
the SVD."""

    @classmethod
    def xvar(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns explained variance ratio for each principal component of the `X` data.
        """
        pca = PCA().fit(X)
        return pca.explained_variance_ratio_

    @classmethod
    def loadings(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns loading for each feature in the data.
        """
        pca = PCA().fit(X)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        df = pd.DataFrame(loadings, columns=[f"PC{i}" for i in range(X.shape[1])], index=X.columns)
        df.loc[:, "Feature"] = df.index
        df = df.loc[:, ["Feature"] + [c for c in df.columns if c != "Feature"]]
        return df

    @classmethod
    def visualize_xvar(self, X: pd.DataFrame, fig: Figure) -> None:
        """Visualize the explained variance and cumulative explained variance by principal components

        Args:
            X: Data set. Shape (n_samples, n_features).
            fig: Matplotlib figure to plot on.
        """
        xvar = self.xvar(X)
        n = xvar.size
        ax = fig.gca()
        ax.plot(range(n), np.cumsum(xvar), label="Cumulative Variance", marker=".")
        bars = ax.bar(range(n), xvar, label="Variance", color="#348ABD")

        # Adding numerical labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height*100:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="dimgray",
            )

        ax.set_xlabel("Principal component")
        ax.set_ylabel("Explained variance ratio")
        ax.set_xticks(range(n))
        ax.grid(False)
        ax.legend()

    @classmethod
    def visualize_loadings_hm(self, X: pd.DataFrame, fig: Figure):
        loadings = PCAModule.loadings(X)
        loadings = loadings.loc[:, ~loadings.columns.isin(["Feature"])]

        ax = fig.gca()
        sns.heatmap(loadings, ax=ax, center=0, cmap="coolwarm")

    @classmethod
    def visualize_loadings(
        self,
        X: pd.DataFrame,
        fig: Figure,
        components: tuple[int],
        topk: int = 3,
    ) -> None:
        """
        Visualizes the PCA-transformed data points using selected principal components.

        Args:
            X: The PCA-transformed DataFrame.
            fig: Matplotlib figure to plot on.
            components: Tuple of two integers representing the selected principal components.
            topk: Number of top features to plot for the selected components.
        """
        assert len(components) == 2, "Exactly two components must be selected for visualization."

        # Extract the selected components
        pc1, pc2 = components

        Xt = PCA().fit_transform(X)

        # Create the plot
        ax = fig.gca()
        ax.scatter(Xt[:, pc1] / Xt[:, pc1].max(), Xt[:, pc2] / Xt[:, pc2].max(), alpha=0.7)

        loadings = PCAModule.loadings(X)

        x = loadings.loc[:, f"PC{pc1}"]
        y = loadings.loc[:, f"PC{pc2}"]

        idx = (x**2 + y**2).sort_values(ascending=False).index[:topk]
        r = np.sqrt(x[idx[0]] ** 2 + y[idx[0]])

        for i in idx:
            ax.annotate("", xy=(x[i] / r, y[i] / r), xytext=(0, 0), arrowprops=dict(color="black", arrowstyle="->"))
            ax.text(x[i] / (2 * r), y[i] / (2 * r), i, color="black")

        # Label the axes
        ax.set_xlabel(f"Principal component {pc1}")
        ax.set_ylabel(f"Principal component {pc2}")

        ax.set_xticks([])
        ax.set_yticks([])
