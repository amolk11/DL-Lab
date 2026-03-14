import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def generate_dataset(n_samples=200, random_state=42):
    """
    Generate a simple 2D linearly separable dataset.
    """
    np.random.seed(random_state)

    X = np.random.randn(n_samples, 2)

    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)

    return X, y


def split_dataset(X, y, test_size=0.2):
    """
    Split dataset into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)


def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    return np.mean(y_true == y_pred)


def plot_dataset(X, y):
    """
    Visualize dataset.
    """
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Dataset Visualization")
    plt.show()


def plot_decision_boundary(model, X, y):
    """
    Plot decision boundary of trained model.
    """

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = np.array([model.predict(x) for x in grid])
    preds = preds.reshape(xx.shape)

    plt.contourf(xx, yy, preds, alpha=0.3, cmap="bwr")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k")
    plt.title("Decision Boundary")
    plt.show()