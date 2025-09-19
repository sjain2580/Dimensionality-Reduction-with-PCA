"""
Project: Visualizing a High-Dimensional Dataset
Author: Your Name Here (for professional portfolio)

Objective: Use Principal Component Analysis (PCA) to reduce the dimensions of a
complex, high-dimensional dataset and visualize its clusters. This script
is designed to be a robust and customizable example for a data science portfolio.

Key Concepts:
- Dimensionality Reduction: A technique for reducing the number of features (variables)
  in a dataset while retaining most of the important information.
- Principal Component Analysis (PCA): An unsupervised learning algorithm that
  transforms a high-dimensional dataset into a new coordinate system where
  the greatest variance by any projection of the data comes to lie on the first
  coordinate (called the first principal component), the second greatest variance
  on the second coordinate, and so on.
- Explained Variance: The proportion of the dataset's total variance that is
  captured by each principal component. This is a crucial metric for evaluating
  how much information is retained after dimensionality reduction.
- Scree Plot: A plot of the eigenvalues (or explained variance) vs. the number of
  principal components, used to determine the optimal number of components to keep.

Dataset: We will use the MNIST handwritten digits dataset, which is a classic
high-dimensional dataset for this type of visualization. Each image is a 28x28
pixel grid, meaning each data point has 784 dimensions (features). We will
reduce this to a lower dimension (2D or 3D) to visualize the inherent structure.

Required Libraries:
- numpy
- scikit-learn
- matplotlib
- seaborn
- argparse (built-in)

To run this script from the command line, use:
python pca_visualization.py --n_components 2 --sample_size 10000
"""

# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
from mpl_toolkits.mplot3d import Axes3D

# Step 2: Set a professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# Step 3: Define the main function to encapsulate the project logic
def main(n_components, sample_size):
    """
    Main function to execute the PCA visualization project.

    Args:
        n_components (int): The number of dimensions to reduce the data to (2 or 3).
        sample_size (int): The number of samples to use from the dataset for faster
                           computation.
    """
    if n_components not in [2, 3]:
        print("Error: The number of components must be either 2 or 3 for visualization.")
        return

    # Step 4: Load and subset the dataset
    print(f"Loading MNIST dataset and subsampling to {sample_size} samples...")
    try:
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X_subset, _, y_subset, _ = train_test_split(X, y, train_size=sample_size, random_state=42, stratify=y)
        print(f"Loaded a subset of {sample_size} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your internet connection and ensure the dataset can be fetched.")
        return

    # Step 5: Data Preprocessing
    # Scaling the data is crucial for PCA as it is sensitive to the scale of features.
    print("Scaling data using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    print("Data scaled successfully.")

    # Step 6: Perform a full PCA analysis to evaluate explained variance
    print("\nPerforming a full PCA to analyze explained variance...")
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # Step 7: Plot the Scree Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
             pca_full.explained_variance_ratio_.cumsum(),
             marker='o', linestyle='-')
    plt.title('Cumulative Explained Variance by Principal Component')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.axvline(x=n_components, color='r', linestyle='--', label=f'Target: {n_components} Components')
    plt.legend()
    plt.savefig('Scree_plot.png')
    plt.show()

    print(f"The first {n_components} components explain "
          f"{pca_full.explained_variance_ratio_[:n_components].sum()*100:.2f}% of the total variance.")

    # Step 8: Apply PCA to reduce dimensions for visualization
    print(f"\nApplying PCA to reduce dimensions to {n_components} for visualization...")
    pca_final = PCA(n_components=n_components)
    X_pca = pca_final.fit_transform(X_scaled)
    print(f"Data reduced from {X_scaled.shape[1]} to {n_components} dimensions.")

    # Step 9: Visualize the PCA-transformed data
    print("Generating the visualization...")
    if n_components == 2:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_subset.astype(int),
                        palette=sns.color_palette("viridis", 10), legend='full',
                        s=50, alpha=0.7)
        plt.title('2D PCA of the MNIST Dataset')
        plt.xlabel(f'Principal Component 1 ({pca_final.explained_variance_ratio_[0]*100:.2f}% Variance)')
        plt.ylabel(f'Principal Component 2 ({pca_final.explained_variance_ratio_[1]*100:.2f}% Variance)')
        plt.legend(title='Digit')
        plt.savefig('Scatter_plot.png')
        plt.show()

    elif n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                             c=y_subset.astype(int), cmap='viridis', s=50, alpha=0.7)
        ax.set_title('3D PCA of the MNIST Dataset')
        ax.set_xlabel(f'PC 1 ({pca_final.explained_variance_ratio_[0]*100:.2f}%)')
        ax.set_ylabel(f'PC 2 ({pca_final.explained_variance_ratio_[1]*100:.2f}%)')
        ax.set_zlabel(f'PC 3 ({pca_final.explained_variance_ratio_[2]*100:.2f}%)')
        plt.colorbar(scatter, label='Digit Label')
        plt.savefig('variance.png')
        plt.show()

    print("\nProject complete. The plots have been displayed.")

# Step 10: Parse command line arguments and run the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a high-dimensional dataset using PCA.")
    parser.add_argument('--n_components', type=int, default=2,
                        help="Number of dimensions to reduce to (2 for 2D plot, 3 for 3D plot).")
    parser.add_argument('--sample_size', type=int, default=10000,
                        help="Number of samples to use for visualization.")
    args = parser.parse_args()

    main(args.n_components, args.sample_size)
