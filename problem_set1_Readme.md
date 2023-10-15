# MNIST Random Walk Model

## Part 1: Load MNIST and Show Montage

```python
# Load MNIST
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from skimage.util import montage
import wandb as wb

# Define GPU functions
def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))

def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))

# Define plot functions
def plot(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='gray')
    ax.axis('off')
    fig.set_size_inches(7, 7)
    plt.show()

def montage_plot(x):
    x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    plot(montage(x))

# Load MNIST dataset
train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

X = train_set.data.numpy()
X_test = test_set.data.numpy()
Y = train_set.targets.numpy()
Y_test = test_set.targets.numpy()

X = X[:, None, :, :] / 255
X_test = X_test[:, None, :, :] / 255

X.shape

montage_plot(X[125:150, 0, :, :])
```

## Part 2: Run random y=mx model on MNIST

```python
# Reshape data
X = X.reshape(X.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

X = GPU_data(X)
Y = GPU_data(Y)
X_test = GPU_data(X_test)
Y_test = GPU_data(Y_test)

X = X.T

x = X[:, 0:64]
M = GPU(np.random.rand(10, 784))
y = M @ x

batch_size = 64
x = X[:, 0:batch_size]
M = GPU(np.random.rand(10, 784))
y = M @ x
y = torch.argmax(y, 0)
accuracy = torch.sum((y == Y[0:batch_size])) / batch_size
```

## Part 3: Train Random Walk Model to at Least 75% Accuracy

```python
m_best = 0
acc_best = 0

for i in range(100000):
    step = 0.0000000001
    m_random = GPU_data(np.random.randn(10, 784))
    m = m_best + step * m_random
    y = m @ X
    y = torch.argmax(y, axis=0)
    acc = ((y == Y)).sum() / len(Y)
    if acc > acc_best:
        print(acc.item())
        m_best = m
        acc_best = acc
```

