import mplcursors
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataset = torch.utils.data.Subset(train_dataset, range(5000))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# 2. Define a 5-layer neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 16)
        self.fc5 = nn.Linear(16, 10)  # output layer

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

model = SimpleNN()

# 3. Define loss and learning rate
criterion = nn.MSELoss()
learning_rate = 0.05


# -------- Gradient Descent Methods --------

def standardGD(model, loss, learning_rate):
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data.add_(-learning_rate * param.grad)
                param.grad.zero_()


def numpySGD(model, images, labels_one_hot, learning_rate):
    """Manual backprop for 5-layer NN with sigmoid+MSE."""
    # Prepare numpy data
    x = images.view(-1, 28*28).numpy()
    y = labels_one_hot.numpy()
    batch_size = x.shape[0]

    # Extract weights & biases
    W1, b1 = model.fc1.weight.data.numpy(), model.fc1.bias.data.numpy()
    W2, b2 = model.fc2.weight.data.numpy(), model.fc2.bias.data.numpy()
    W3, b3 = model.fc3.weight.data.numpy(), model.fc3.bias.data.numpy()
    W4, b4 = model.fc4.weight.data.numpy(), model.fc4.bias.data.numpy()
    W5, b5 = model.fc5.weight.data.numpy(), model.fc5.bias.data.numpy()

    def sigmoid(z): return 1 / (1 + np.exp(-z))

    # -------- Forward Pass --------
    z1 = x @ W1.T + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2.T + b2
    a2 = sigmoid(z2)
    z3 = a2 @ W3.T + b3
    a3 = sigmoid(z3)
    z4 = a3 @ W4.T + b4
    a4 = sigmoid(z4)
    z5 = a4 @ W5.T + b5
    a5 = sigmoid(z5)

    loss = np.mean((a5 - y) ** 2)

    # -------- Backpropagation --------
    dA5 = 2 * (a5 - y) / batch_size
    dZ5 = dA5 * a5 * (1 - a5)
    dW5 = dZ5.T @ a4
    db5 = np.sum(dZ5, axis=0)

    dA4 = dZ5 @ W5
    dZ4 = dA4 * a4 * (1 - a4)
    dW4 = dZ4.T @ a3
    db4 = np.sum(dZ4, axis=0)

    dA3 = dZ4 @ W4
    dZ3 = dA3 * a3 * (1 - a3)
    dW3 = dZ3.T @ a2
    db3 = np.sum(dZ3, axis=0)

    dA2 = dZ3 @ W3
    dZ2 = dA2 * a2 * (1 - a2)
    dW2 = dZ2.T @ a1
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ W2
    dZ1 = dA1 * a1 * (1 - a1)
    dW1 = dZ1.T @ x
    db1 = np.sum(dZ1, axis=0)

    # -------- Gradient Descent Update --------
    W5 -= learning_rate * dW5; b5 -= learning_rate * db5
    W4 -= learning_rate * dW4; b4 -= learning_rate * db4
    W3 -= learning_rate * dW3; b3 -= learning_rate * db3
    W2 -= learning_rate * dW2; b2 -= learning_rate * db2
    W1 -= learning_rate * dW1; b1 -= learning_rate * db1

    # Copy back to model
    model.fc1.weight.data.copy_(torch.from_numpy(W1))
    model.fc1.bias.data.copy_(torch.from_numpy(b1))
    model.fc2.weight.data.copy_(torch.from_numpy(W2))
    model.fc2.bias.data.copy_(torch.from_numpy(b2))
    model.fc3.weight.data.copy_(torch.from_numpy(W3))
    model.fc3.bias.data.copy_(torch.from_numpy(b3))
    model.fc4.weight.data.copy_(torch.from_numpy(W4))
    model.fc4.bias.data.copy_(torch.from_numpy(b4))
    model.fc5.weight.data.copy_(torch.from_numpy(W5))
    model.fc5.bias.data.copy_(torch.from_numpy(b5))

    return loss


# 4. Training loop
losses = []
for epoch in range(1000):
    epoch_loss = 0.0
    batch_count = 0
    
    for images, labels in train_loader:
        outputs = model(images)
        labels_one_hot = torch.zeros(labels.size(0), 10)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
        loss = criterion(outputs, labels_one_hot)
        
        numpySGD(model, images, labels_one_hot, learning_rate)
        
        epoch_loss += loss.item()
        batch_count += 1
        losses.append(loss.item())
    
    avg_epoch_loss = epoch_loss / batch_count
    print(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.6f}")


# 5. Evaluate on test set
print("\n" + "="*50)
print("EVALUATION ON TEST SET")
print("="*50)

model.eval()
correct = 0
total = 0
test_losses = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        labels_one_hot = torch.zeros(labels.size(0), 10)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
        test_loss = criterion(outputs, labels_one_hot)
        test_losses.append(test_loss.item())
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
avg_test_loss = sum(test_losses) / len(test_losses)

print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Average Test Loss: {avg_test_loss:.6f}")
print(f"Correct Predictions: {correct}/{total}")
print("="*50)

# 6. Plot loss graph
plt.figure(figsize=(12, 8))
line, = plt.plot(losses, label='Training Loss', markersize=1)
mplcursors.cursor([line], hover=True)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
