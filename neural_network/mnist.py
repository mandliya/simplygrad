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
    # use a small sample of the data to test the model
    train_images = train_images[:10000]
    train_labels = train_labels[:10000]
    test_images = test_images[:1000]
    test_labels = test_labels[:1000]
    batch_size = 32
    epochs = 2
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i in range(0, len(train_images), batch_size):
            images = train_images[i:i+batch_size].numpy()
            labels = train_labels[i:i+batch_size].numpy()
            images = Tensor(images, requires_grad=False)
            labels = Tensor(labels, requires_grad=False)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_images)//batch_size}, Loss: {loss.item()}")
            optimizer.step()
    print("Training complete")
    test_images_t = Tensor(test_images.numpy(), requires_grad=False)
    test_labels_t = Tensor(test_labels.numpy(), requires_grad=False)
    test_outputs = model(test_images_t)
    test_loss = criterion(test_outputs, test_labels_t).item()
    print(f"Test loss: {test_loss}")
    test_preds = xp.argmax(test_outputs.data, axis=1)
    test_accuracy = float(xp.mean((test_preds == test_labels_t.data).astype(float)))
    print(f"Test accuracy: {test_accuracy:.4f}")