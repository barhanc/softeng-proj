import numpy as np
import pandas as pd
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
            )

        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_xticks(range(n))
        ax.legend()

    @classmethod
    def visualize_selected_components(self, pca_transformed, loadings, components, top_n_features=5):
        """
        Visualizes the PCA-transformed data points using selected principal components.

        Parameters:
        pca_transformed (pd.DataFrame): The PCA-transformed DataFrame.
        loadings (pd.DataFrame): The PCA loadings.
        components (list): List of two integers representing the selected principal components (e.g., [1, 3]).
        top_n_features (int): Number of top features to plot for the selected components.

        Returns:
        fig: A matplotlib figure object with the plot.
        """
        if len(components) != 2:
            raise ValueError("Exactly two components must be selected for visualization.")

        # Extract the selected components
        pc1, pc2 = components

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(pca_transformed[f"pca{pc1}"], pca_transformed[f"pca{pc2}"], alpha=0.7)

        # Label the axes
        ax.set_xlabel(f"Principal Component {pc1}")
        ax.set_ylabel(f"Principal Component {pc2}")
        ax.set_title(f"PCA: PC{pc1} vs PC{pc2}")

        # Plot the top_n_features for the selected components as vectors
        for i in range(top_n_features):
            feature_name_pc1 = loadings[f"PC{pc1}"].abs().sort_values(ascending=False).index[i]
            feature_name_pc2 = loadings[f"PC{pc2}"].abs().sort_values(ascending=False).index[i]

            # Vector for PC1
            vector_pc1 = loadings.loc[feature_name_pc1, [f"PC{pc1}", f"PC{pc2}"]]
            ax.arrow(
                0,
                0,
                vector_pc1[f"PC{pc1}"],
                vector_pc1[f"PC{pc2}"],
                color="r",
                alpha=0.5,
                head_width=0.05,
                head_length=0.1,
            )
            ax.text(vector_pc1[f"PC{pc1}"] * 1.1, vector_pc1[f"PC{pc2}"] * 1.1, feature_name_pc1, color="r")

            # Vector for PC2
            if feature_name_pc2 != feature_name_pc1:
                vector_pc2 = loadings.loc[feature_name_pc2, [f"PC{pc1}", f"PC{pc2}"]]
                ax.arrow(
                    0,
                    0,
                    vector_pc2[f"PC{pc1}"],
                    vector_pc2[f"PC{pc2}"],
                    color="b",
                    alpha=0.5,
                    head_width=0.05,
                    head_length=0.1,
                )
                ax.text(vector_pc2[f"PC{pc1}"] * 1.1, vector_pc2[f"PC{pc2}"] * 1.1, feature_name_pc2, color="b")

        plt.grid()
        plt.axhline(0, color="grey", lw=0.5)
        plt.axvline(0, color="grey", lw=0.5)
        plt.tight_layout()

        return fig
