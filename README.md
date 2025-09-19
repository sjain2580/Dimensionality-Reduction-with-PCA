# Dimensionality Reduction with PCA

## Overview

This project demonstrates the use of Principal Component Analysis (PCA) to reduce the dimensionality of a complex dataset and visualize its inherent structure. It is a practical example of unsupervised learning and a key technique in data science for exploratory data analysis and feature engineering.

## Features

- Dimensionality Reduction: Uses PCA to transform a high-dimensional dataset into a 2D or 3D space.

- Explained Variance Analysis: Generates a scree plot to show the cumulative variance explained by each principal component, helping to determine the optimal number of components to retain.

- Interactive Visualization: Creates scatter plots (2D and 3D) of the transformed data, colored by their original labels to visualize how well PCA separates the classes.

- Customizable: Allows you to specify the number of components and the dataset sample size from the command line.

## Technologies Used

- Python: The core programming language for the project.

- Scikit-learn: The primary machine learning library for implementing PCA.

- NumPy: Used for numerical operations and efficient handling of data arrays.

- Matplotlib: A powerful plotting library used for all visualizations.

- Seaborn: Built on Matplotlib, used for creating visually appealing statistical plots.

## Data Analysis & Processing

The project utilizes the MNIST handwritten digits dataset, which consists of 784-dimensional images. Before applying PCA, the data undergoes two crucial preprocessing steps:

1. Subsampling: A subset of the data is used to ensure faster processing.
2. Standard Scaling: The data is scaled using StandardScaler to have a mean of 0 and a standard deviation of 1. This is a critical step because PCA is sensitive to the scale of features.

## Model Used

The core of this project is Principal Component Analysis (PCA). PCA is an unsupervised learning algorithm that finds the directions of maximum variance in the data. These new directions, called principal components, are used to project the data into a lower-dimensional space while preserving as much variance (information) as possible.

## Model Training

PCA is not "trained" in the traditional sense. Instead, it is fitted to the scaled data using the .fit() method. This process identifies the principal components and the amount of variance they explain. The .fit_transform() method then applies this transformation to the data, projecting it onto the selected number of components.

## How to Run the Project

1. Clone the repository:

```bash
git clone <https://github.com/sjain2580/Loan-Approval-Prediction-with-Random-Forest>
cd <repository_name>
```

2. Create and activate a virtual environment (optional but recommended):python -m venv venv

- On Windows:
  
```bash
.\venv\Scripts\activate
```

- On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Script:

```bash
python pca_visualization.py
```

For 2D Visualization:

```bash
python pca_visualization.py --n_components 2 --sample_size 10000
```

For 3D Visualization:

```bash
python pca_visualization.py --n_components 3 --sample_size 10000
```

## Visualization

The project generates two key plots:

- Scree Plot: This plot displays the cumulative explained variance. It helps you see how much information is retained as you increase the number of principal components.2D/3D.

- Scatter Plot: This is the final visualization of the transformed data. Each point represents a handwritten digit, and the colors correspond to their original class labels. This plot reveals if the digits form distinct, separable clusters in the lower-dimensional space.

## Contributors

**<https://github.com/sjain2580>**
Feel free to fork this repository, submit issues, or pull requests to improve the project. Suggestions for model enhancement or additional visualizations are welcome!

## Connect with Me

Feel free to reach out if you have any questions or just want to connect!
**[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sjain04/)**
**[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/sjain2580)**
**[![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:sjain040395@gmail.com)**

---
