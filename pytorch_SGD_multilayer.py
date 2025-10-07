import mplcursors
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time


# grab the MNIST data
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataset = torch.utils.data.Subset(train_dataset, range(5000))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 5-layer network (4 hidden layers + output)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

def create_model_with_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return SimpleNN()

def standardGD(model, loss, learning_rate):
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data.add_(-learning_rate * param.grad)
                param.grad.zero_()

def pytorchGD(model, images, labels_one_hot, learning_rate):
    # STANDARD GD - updates both layers at the same time
    # same backprop setup as pytorchSGD but different update order
    
    # convert to pytorch
    x = images.view(-1, 28*28)
    y = labels_one_hot

    # pull out weights/biases
    W1 = model.fc1.weight.data.clone()
    b1 = model.fc1.bias.data.clone()
    W2 = model.fc2.weight.data.clone()
    b2 = model.fc2.bias.data.clone()
    W3 = model.fc3.weight.data.clone()
    b3 = model.fc3.bias.data.clone()
    W4 = model.fc4.weight.data.clone()
    b4 = model.fc4.bias.data.clone()
    W5 = model.fc5.weight.data.clone()
    b5 = model.fc5.bias.data.clone()

    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))
    
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
    
    loss = torch.mean((a5 - y) ** 2)

    batch_size = x.shape[0]
    dA5 = 2 * (a5 - y) / (batch_size * 10)
    dZ5 = dA5 * a5 * (1 - a5)

    dW5 = dZ5.T @ a4
    db5 = torch.sum(dZ5, dim=0)

    dA4 = dZ5 @ W5
    dZ4 = dA4 * a4 * (1 - a4)
    dW4 = dZ4.T @ a3
    db4 = torch.sum(dZ4, dim=0)

    dA3 = dZ4 @ W4
    dZ3 = dA3 * a3 * (1 - a3)
    dW3 = dZ3.T @ a2
    db3 = torch.sum(dZ3, dim=0)

    dA2 = dZ3 @ W3
    dZ2 = dA2 * a2 * (1 - a2)
    dW2 = dZ2.T @ a1
    db2 = torch.sum(dZ2, dim=0)

    dA1 = dZ2 @ W2
    dZ1 = dA1 * a1 * (1 - a1)
    dW1 = dZ1.T @ x
    db1 = torch.sum(dZ1, dim=0)

    W5 -= learning_rate * dW5
    b5 -= learning_rate * db5
    W4 -= learning_rate * dW4
    b4 -= learning_rate * db4
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    model.fc1.weight.data.copy_(W1)
    model.fc1.bias.data.copy_(b1)
    model.fc2.weight.data.copy_(W2)
    model.fc2.bias.data.copy_(b2)
    model.fc3.weight.data.copy_(W3)
    model.fc3.bias.data.copy_(b3)
    model.fc4.weight.data.copy_(W4)
    model.fc4.bias.data.copy_(b4)
    model.fc5.weight.data.copy_(W5)
    model.fc5.bias.data.copy_(b5)

    return loss.item()



def pytorchSGD(model, images, labels_one_hot, learning_rate):
    # SEQUENTIAL GD - updates layer 2 first then layer 1
    # same backprop setup as pytorchGD but different update order
    
    # convert to pytorch
    x = images.view(-1, 28*28)
    y = labels_one_hot

    W1 = model.fc1.weight.data.clone()
    b1 = model.fc1.bias.data.clone()
    W2 = model.fc2.weight.data.clone()
    b2 = model.fc2.bias.data.clone()
    W3 = model.fc3.weight.data.clone()
    b3 = model.fc3.bias.data.clone()
    W4 = model.fc4.weight.data.clone()
    b4 = model.fc4.bias.data.clone()
    W5 = model.fc5.weight.data.clone()
    b5 = model.fc5.bias.data.clone()

    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))
    
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
    
    loss = torch.mean((a5 - y) ** 2)

    batch_size = x.shape[0]
    dA5 = 2 * (a5 - y) / (batch_size * 10)
    dZ5 = dA5 * a5 * (1 - a5)

    dW5 = dZ5.T @ a4
    db5 = torch.sum(dZ5, dim=0)

    W5 -= learning_rate * dW5
    b5 -= learning_rate * db5

    dA4 = dZ5 @ W5
    dZ4 = dA4 * a4 * (1 - a4)
    dW4 = dZ4.T @ a3
    db4 = torch.sum(dZ4, dim=0)

    W4 -= learning_rate * dW4
    b4 -= learning_rate * db4

    dA3 = dZ4 @ W4
    dZ3 = dA3 * a3 * (1 - a3)
    dW3 = dZ3.T @ a2
    db3 = torch.sum(dZ3, dim=0)

    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    dA2 = dZ3 @ W3
    dZ2 = dA2 * a2 * (1 - a2)
    dW2 = dZ2.T @ a1
    db2 = torch.sum(dZ2, dim=0)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    dA1 = dZ2 @ W2
    dZ1 = dA1 * a1 * (1 - a1)
    dW1 = dZ1.T @ x
    db1 = torch.sum(dZ1, dim=0)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    model.fc1.weight.data.copy_(W1)
    model.fc1.bias.data.copy_(b1)
    model.fc2.weight.data.copy_(W2)
    model.fc2.bias.data.copy_(b2)
    model.fc3.weight.data.copy_(W3)
    model.fc3.bias.data.copy_(b3)
    model.fc4.weight.data.copy_(W4)
    model.fc4.bias.data.copy_(b4)
    model.fc5.weight.data.copy_(W5)
    model.fc5.bias.data.copy_(b5)

    return loss.item()



def compare_methods(seed=42, epochs=100, learning_rate=0.01):
    start_time = time.time()
    model1 = create_model_with_seed(seed)
    model2 = create_model_with_seed(seed)
    model3 = create_model_with_seed(seed)
    
    criterion = nn.MSELoss()
    losses1 = []
    losses2 = []
    losses3 = []
    
    print(f"Comparing methods with seed {seed}")
    print("="*50)
    
    print("Training with normalGD...")
    normalGD_start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for images, labels in train_loader:
            outputs = model1(images)
            labels_one_hot = torch.zeros(labels.size(0), 10)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
            loss = criterion(outputs, labels_one_hot)
            standardGD(model1, loss, learning_rate)
            epoch_loss += loss.item()
            batch_count += 1
            losses1.append(loss.item())
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.6f}")
    normalGD_time = time.time() - normalGD_start
    
    print("\nTraining with pytorchGD...")
    pytorchGD_start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for images, labels in train_loader:
            outputs = model2(images)
            labels_one_hot = torch.zeros(labels.size(0), 10)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
            loss = criterion(outputs, labels_one_hot)
            pytorchGD(model2, images, labels_one_hot, learning_rate)
            epoch_loss += loss.item()
            batch_count += 1
            losses2.append(loss.item())
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.6f}")
    pytorchGD_time = time.time() - pytorchGD_start
    
    print("\nTraining with pytorchSGD...")
    pytorchSGD_start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for images, labels in train_loader:
            outputs = model3(images)
            labels_one_hot = torch.zeros(labels.size(0), 10)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
            loss = criterion(outputs, labels_one_hot)
            pytorchSGD(model3, images, labels_one_hot, learning_rate)
            epoch_loss += loss.item()
            batch_count += 1
            losses3.append(loss.item())
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.6f}")
    pytorchSGD_time = time.time() - pytorchSGD_start
    
    print("\n" + "="*50)
    print("EVALUATION ON TEST SET")
    print("="*50)
    accuracies = []
    for i, model in enumerate([model1, model2, model3]):
        method_name = "normalGD" if i == 0 else "pytorchGD" if i == 1 else "pytorchSGD"
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
        accuracies.append(test_accuracy)
        print(f"{method_name} - Test Accuracy: {test_accuracy:.2f}%")
        print(f"{method_name} - Average Test Loss: {avg_test_loss:.6f}")
        print(f"{method_name} - Correct Predictions: {correct}/{total}")
        if i == 0:
            print(f"{method_name} - Training Time: {normalGD_time:.2f} seconds")
        elif i == 1:
            print(f"{method_name} - Training Time: {pytorchGD_time:.2f} seconds")
        else:
            print(f"{method_name} - Training Time: {pytorchSGD_time:.2f} seconds")
        print("-" * 30)
    print("="*50)
    
    plt.figure(figsize=(12, 8))
    line1, = plt.plot(losses1[50:], label='normalGD Loss', color='blue', markersize=1)
    line2, = plt.plot(losses2[50:], label='pytorchGD Loss', color='red', markersize=1)
    line3, = plt.plot(losses3[50:], label='pytorchSGD Loss', color='green', markersize=1)
    mplcursors.cursor([line1, line2, line3], hover=True)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison: normalGD vs pytorchGD vs pytorchSGD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    folderName = "pytorch_five_layer_testing1"
    os.makedirs(folderName, exist_ok=True)
    base_filename = f'{folderName}/seed_{seed}_epochs_{epochs}_lr_{learning_rate}'
    filename = f'{base_filename}.png'
    counter = 1
    while os.path.exists(filename):
        filename = f'{base_filename}({counter}).png'
        counter += 1
    plt.savefig(filename)

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    for i in range(1):
        compare_methods(seed=i, epochs=50, learning_rate=0.1)
