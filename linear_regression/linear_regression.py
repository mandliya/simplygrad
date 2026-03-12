"""
Linear Regression on a Real Dataset
We will use deeplygrad to implement linear regression on a real dataset. 
The dataset is the California housing dataset, which contains information about 
housing prices in California. 
We will use this dataset to train a linear regression model to predict housing prices based on 
various features.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deeplygrad import Tensor, xp, BACKEND_NAME
print(f"Using backend: {BACKEND_NAME}")

def load_data():
    # Load the California housing dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

def train_linear_regression(X_train, y_train, X_test, y_test, lr=0.01, epochs=100):
    n_samples, n_features = X_train.shape

    # Convert data to Tensors
    X_train = Tensor(X_train, requires_grad=False)
    y_train = Tensor(y_train, requires_grad=False)
    X_test = Tensor(X_test, requires_grad=False)
    y_test = Tensor(y_test, requires_grad=False)

    # Initialize weights and bias
    W = Tensor(xp.random.randn(n_features), requires_grad=True)
    b = Tensor(0.0, requires_grad=True)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        W.zero_grad()
        b.zero_grad()
        y_pred_train = X_train @ W + b
        train_loss = ((y_pred_train - y_train) ** 2).mean()
        train_loss.backward()
        W.data -= lr * W.grad
        b.data -= lr * b.grad
        train_loss = train_loss.item()
        train_losses.append(train_loss)
        y_pred_test = X_test @ W + b
        test_loss = ((y_pred_test - y_test) ** 2).mean()
        test_loss = test_loss.item()
        test_losses.append(test_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return W, b, train_losses, test_losses

def plot_results(train_losses, test_losses, X_test, y_test, W, b):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training and Test Loss over Epochs')
    ax1.legend()
    ax1.grid()

    y_pred = (Tensor(X_test) @ W + b).numpy()
    ax2.scatter(y_test, y_pred, alpha=0.3, s=10)
    lims = [0, max(y_test.max(), y_pred.max())]
    ax2.plot(lims, lims, 'r--', label='Ideal')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Test Set: Predicted vs Actual')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.savefig('results_plot.png')
    print("Plot saved as 'results_plot.png'")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    W, b, train_losses, test_losses = train_linear_regression(X_train, y_train, X_test, y_test)
    plot_results(train_losses, test_losses, X_test, y_test, W, b)