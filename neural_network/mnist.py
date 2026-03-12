"""
MNIST using deeplygrad

We'll train a neural network to classify MNIST digits using the
deeplygrad library.

3-layer MLP: 784 -> 128 -> 64 -> 10
Activation: ReLU
Loss: Cross-Entropy Loss
Optimizer: Adam

Dataset: PyTorch's built-in MNIST
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision.datasets import MNIST
from deeplygrad.nn import Module, Linear, ReLU, CrossEntropyLoss
from deeplygrad.optim import Adam
from deeplygrad.backend import xp, BACKEND_NAME
from deeplygrad.tensor import Tensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

print(f"Using backend: {BACKEND_NAME}")

def get_data():
    train_dataset = MNIST(root='./data', train=True, download=True)
    test_dataset = MNIST(root='./data', train=False, download=True)
    return train_dataset, test_dataset

def preprocess_data(dataset):
    images = dataset.data
    labels = dataset.targets
    images = images.reshape(-1, 28 * 28) / 255.0
    labels = labels.long()
    return images, labels


class MLP(Module):
    """
    3-layer MLP: 784 -> 128 -> 64 -> 10
    Activation: ReLU
    Loss: Cross-Entropy Loss
    Optimizer: Adam
    """
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.relu1 = ReLU()
        self.fc2 = Linear(128, 64)
        self.relu2 = ReLU()
        self.fc3 = Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    xp.random.seed(42)
    train_dataset, test_dataset = get_data()
    train_images, train_labels = preprocess_data(train_dataset)
    test_images, test_labels = preprocess_data(test_dataset)
    model = MLP()
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    # Split: 8000 train, 2000 val from training set; test set separate
    val_images = train_images[8000:10000]
    val_labels = train_labels[8000:10000]
    train_images = train_images[:8000]
    train_labels = train_labels[:8000]
    test_images = test_images[:1000]
    test_labels = test_labels[:1000]
    batch_size = 32
    epochs = 5
    epoch_train_losses = []
    epoch_val_losses = []
    batch_losses = []

    test_images_t = Tensor(test_images.numpy(), requires_grad=False)
    test_labels_t = Tensor(test_labels.numpy(), requires_grad=False)
    val_images_t = Tensor(val_images.numpy(), requires_grad=False)
    val_labels_t = Tensor(val_labels.numpy(), requires_grad=False)
    for epoch in range(epochs):
        epoch_batch_losses = []
        for i in range(0, len(train_images), batch_size):
            images = train_images[i:i+batch_size].numpy()
            labels = train_labels[i:i+batch_size].numpy()
            images = Tensor(images, requires_grad=False)
            labels = Tensor(labels, requires_grad=False)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            l = loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i//batch_size + 1}/{len(train_images)//batch_size}, Loss: {l:.4f}")
            epoch_batch_losses.append(l)
            batch_losses.append(l)

        avg_train_loss = np.mean(epoch_batch_losses)
        epoch_train_losses.append(avg_train_loss)

        val_outputs = model(val_images_t)
        val_loss = criterion(val_outputs, val_labels_t).item()
        epoch_val_losses.append(val_loss)

        val_preds = xp.argmax(val_outputs.data, axis=1)
        val_accuracy = float(xp.mean((val_preds == val_labels_t.data).astype(float)))

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    print("Training complete")
    test_outputs = model(test_images_t)
    test_preds = xp.argmax(test_outputs.data, axis=1)
    test_accuracy = float(xp.mean((test_preds == test_labels_t.data).astype(float)))
    print(f"Final test accuracy: {test_accuracy:.4f}")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(batch_losses, alpha=0.3, label='Batch Loss')
    window = max(1, len(batch_losses) // 50)
    smoothed = np.convolve(batch_losses, np.ones(window)/window, mode='valid')
    ax1.plot(smoothed, color='red', label=f'Smoothed (window={window})')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss per Batch')
    ax1.legend()
    ax1.grid()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, epochs + 1), epoch_train_losses, 'o-', label='Train')
    ax2.plot(range(1, epochs + 1), epoch_val_losses, 's--', label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Avg Loss')
    ax2.set_title('Train vs Validation Loss per Epoch')
    ax2.legend()
    ax2.grid()

    ax_grid = fig.add_subplot(gs[1, :])
    ax_grid.set_axis_off()
    ax_grid.set_title('Test Predictions (red = wrong)', fontsize=14, pad=10)
    rows, cols = 3, 8
    indices = np.random.choice(len(test_images), rows * cols, replace=False)
    inner_gs = gs[1, :].subgridspec(rows, cols, wspace=0.3, hspace=0.5)
    for idx_i, sample_idx in enumerate(indices):
        r, c = divmod(idx_i, cols)
        ax = fig.add_subplot(inner_gs[r, c])
        img = test_images[sample_idx].numpy().reshape(28, 28)
        pred = int(test_preds[sample_idx])
        actual = int(test_labels_t.data[sample_idx])
        ax.imshow(img, cmap='gray')
        color = 'green' if pred == actual else 'red'
        ax.set_title(f'P:{pred} A:{actual}', fontsize=8, color=color)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('neural_network/mnist_loss.png', dpi=150)
    print("Plot saved as 'neural_network/mnist_loss.png'")