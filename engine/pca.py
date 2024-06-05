import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


class PCAModule:
    @classmethod
    def pca(self, X: pd.DataFrame):
        pass

    @classmethod
    def perform_pca(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Perform Principal Component Analysis (PCA) on the given DataFrame.

        Parameters:
        X (pd.DataFrame): The input DataFrame with features as columns.

        Returns:
        principal_components (np.ndarray): The transformed data in the principal component space.
        explained_variance (np.ndarray): The explained variance ratio for each principal component.
        loadings (pd.DataFrame): The loadings matrix where each column represents a principal component
                                 and each row represents the contribution of the original feature to that component.
        """

        size = len(X.columns)
        pca = PCA()

        principal_components = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_
        loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(size)], index=X.columns)

        return principal_components, explained_variance, loadings

    @classmethod
    def plot_explained_variance(self, explained_variance):
        """
        Creates a bar plot to visualize the explained variance and cumulative explained variance by principal components.

        Parameters:
        explained_variance (np.ndarray): Array containing the explained variance ratio for each principal component.

        Returns:
        plt.Figure: A matplotlib figure object with the plot.
        """

        fig, ax = plt.subplots()
        n_components = len(explained_variance)

        ax.bar(range(1, n_components + 1), np.cumsum(explained_variance), label="Cumulative Variance", color="green")
        bars = ax.bar(range(1, n_components + 1), explained_variance, label="Variance", color="red")

        # Adding numerical labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        ax.set_xlabel("Principal Components")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("Explained Variance by Principal Components")
        ax.set_xticks(range(1, n_components + 1))
        ax.legend()
        return fig

    @classmethod
    def most_impactful_features(self, loadings, top_n=5):
        """
        Identifies the most impactful features for each principal component.

        Parameters:
        loadings (pd.DataFrame): Loadings of the PCA components.
        top_n (int): Number of top features to select for each principal component.

        Returns:
        pd.DataFrame: A DataFrame with the most impactful features for each principal component.
        """
        impactful_features = pd.DataFrame()

        for col in loadings.columns:
            # Take the absolute value of the loadings to find the most impactful features
            sorted_features = loadings[col].abs().sort_values(ascending=False).head(top_n)
            impactful_features[col] = sorted_features.index

        return impactful_features

    @classmethod
    def plot_impactful_features(self, loadings):
        """
        Creates a bar plot to visualize the magnitude of the loadings for each feature.

        Parameters:
        loadings (pd.DataFrame): Loadings of the PCA components.

        Returns:
        fig: A matplotlib figure object with the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        loadings_plot = loadings.abs().sort_values(by=loadings.columns[0], ascending=False)

        loadings_plot.plot(kind="bar", ax=ax, colormap="viridis")

        ax.set_title("Magnitude of Loadings for Each Feature")
        ax.set_xlabel("Features")
        ax.set_ylabel("Loading Magnitude")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        return fig

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

    @classmethod
    def plot_impactful_features_v2(self, loadings, top_n_features=5):
        """
        Creates a bar plot to visualize the impact of each feature on the principal components.

        Parameters:
        loadings (pd.DataFrame): Loadings of the PCA components.
        top_n_features (int): Number of top features to plot for each principal component.

        Returns:
        fig: A matplotlib figure object with the plot.
        """
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the impact of each feature on the principal components
        for i, component in enumerate(loadings.columns):
            # Get the top n features for this component, sorted by absolute impact
            sorted_loadings = loadings[component].abs().sort_values(ascending=False).head(top_n_features)
            sorted_loadings = loadings.loc[sorted_loadings.index, component]

            # Create a bar plot for the current component
            ax.bar(
                [f"{feature} ({component})" for feature in sorted_loadings.index],
                sorted_loadings.values,
                label=f"PC{i+1}",
            )

        # Set the labels and title
        ax.set_xlabel("Principal Components and Features")
        ax.set_ylabel("Loading Magnitude")
        ax.set_title("Impact of Features on Principal Components")
        ax.legend()
        plt.xticks(rotation=90)
        plt.tight_layout()

        return fig
