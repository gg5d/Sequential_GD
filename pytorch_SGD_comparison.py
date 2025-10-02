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

# train_dataset = torch.utils.data.Subset(train_dataset, range(5000))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# just a basic 2-layer network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # layer 1
        self.fc2 = nn.Linear(128, 10)     # 10 outputs for digits 0-9
    def forward(self, x):
        x = x.view(-1, 28*28)        # squash 28x28 down to 784
        x = torch.sigmoid(self.fc1(x))  # run through sigmoid (seems to work better)
        x = torch.sigmoid(self.fc2(x))  # another sigmoid
        return x

def create_model_with_seed(seed):
    # makes a model with same starting weights every time (for fair comparison)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return SimpleNN()

def standardGD(model, loss, learning_rate):
    # do backprop
    loss.backward()
    
    # update weights using gradients (vectorized way)
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                # doing it in-place to save memory
                param.data.add_(-learning_rate * param.grad)
                param.grad.zero_()

def pytorchGD(model, images, labels_one_hot, learning_rate):
    # manual backprop with PyTorch tensors; update ALL layers together (standard GD)
    with torch.no_grad():
        x = images.view(-1, 28*28)                 # (B, 784)
        y = labels_one_hot                         # (B, 10)

        W1 = model.fc1.weight.detach().clone()     # (H, 784)
        b1 = model.fc1.bias.detach().clone()       # (H,)
        W2 = model.fc2.weight.detach().clone()     # (10, H)
        b2 = model.fc2.bias.detach().clone()       # (10,)

        def sigmoid(t): return 1.0 / (1.0 + torch.exp(-t))

        z1 = x @ W1.t() + b1                       # (B, H)
        a1 = sigmoid(z1)                           # (B, H)
        z2 = a1 @ W2.t() + b2                      # (B, 10)
        a2 = sigmoid(z2)                           # (B, 10)

        # MSE(mean) matching: divide by batch_size * 10
        B = x.shape[0]
        dA2 = 2.0 * (a2 - y) / (B * 10)

        dZ2 = dA2 * a2 * (1.0 - a2)                # (B, 10)
        dW2 = dZ2.t() @ a1                         # (10, H)
        db2 = dZ2.sum(dim=0)                       # (10,)

        dA1 = dZ2 @ W2                             # (B, H)
        dZ1 = dA1 * a1 * (1.0 - a1)                # (B, H)
        dW1 = dZ1.t() @ x                          # (H, 784)
        db1 = dZ1.sum(dim=0)                       # (H,)

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        model.fc1.weight.copy_(W1)
        model.fc1.bias.copy_(b1)
        model.fc2.weight.copy_(W2)
        model.fc2.bias.copy_(b2)

    # return a scalar loss value to match original behavior
    loss_val = torch.mean((a2 - y) ** 2).item()
    return loss_val


def pytorchSGD(model, images, labels_one_hot, learning_rate):
    # manual backprop with PyTorch tensors; SEQUENTIAL: update layer 2, then layer 1 (using updated W2)
    with torch.no_grad():
        x = images.view(-1, 28*28)                 # (B, 784)
        y = labels_one_hot                         # (B, 10)

        W1 = model.fc1.weight.detach().clone()     # (H, 784)
        b1 = model.fc1.bias.detach().clone()       # (H,)
        W2 = model.fc2.weight.detach().clone()     # (10, H)
        b2 = model.fc2.bias.detach().clone()       # (10,)

        def sigmoid(t): return 1.0 / (1.0 + torch.exp(-t))

        z1 = x @ W1.t() + b1                       # (B, H)
        a1 = sigmoid(z1)                           # (B, H)
        z2 = a1 @ W2.t() + b2                      # (B, 10)
        a2 = sigmoid(z2)                           # (B, 10)

        # MSE(mean) matching: divide by batch_size * 10
        B = x.shape[0]
        dA2 = 2.0 * (a2 - y) / (B * 10)

        dZ2 = dA2 * a2 * (1.0 - a2)                # (B, 10)
        dW2 = dZ2.t() @ a1                         # (10, H)
        db2 = dZ2.sum(dim=0)                       # (10,)

        # UPDATE LAYER 2 FIRST
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        # backprop to layer 1 using UPDATED W2
        dA1 = dZ2 @ W2                             # (B, H)
        dZ1 = dA1 * a1 * (1.0 - a1)                # (B, H)
        dW1 = dZ1.t() @ x                          # (H, 784)
        db1 = dZ1.sum(dim=0)                       # (H,)

        # then update layer 1
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        model.fc1.weight.copy_(W1)
        model.fc1.bias.copy_(b1)
        model.fc2.weight.copy_(W2)
        model.fc2.bias.copy_(b2)

    # return a scalar loss value to match original behavior
    loss_val = torch.mean((a2 - y) ** 2).item()
    return loss_val


def compare_methods(seed=42, epochs=100, learning_rate=0.01):
    # train 3 models with different methods and compare them
    
    # start timing
    start_time = time.time()
    
    # make 3 identical models
    model1 = create_model_with_seed(seed)
    model2 = create_model_with_seed(seed)
    model3 = create_model_with_seed(seed)
    
    criterion = nn.MSELoss()
    losses1 = []
    losses2 = []
    losses3 = []
    
    print(f"Comparing methods with seed {seed}")
    print("="*50)
    
    # train with regular gradient descent
    print("Training with normalGD...")
    normalGD_start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for images, labels in train_loader:
            # run forward
            outputs = model1(images)
            # need one-hot for MSE loss
            labels_one_hot = torch.zeros(labels.size(0), 10)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
            loss = criterion(outputs, labels_one_hot)
            
            # backprop and update
            standardGD(model1, loss, learning_rate)
            
            epoch_loss += loss.item()
            batch_count += 1
            losses1.append(loss.item())
        
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.6f}")
    
    normalGD_time = time.time() - normalGD_start
    
    # train with numpy gradient descent
    print("\nTraining with numpyGD...")
    numpyGD_start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for images, labels in train_loader:
            # forward pass
            outputs = model2(images)
            # convert to one-hot
            labels_one_hot = torch.zeros(labels.size(0), 10)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
            loss = criterion(outputs, labels_one_hot)
            
            # our custom numpy backprop
            numpyGD(model2, images, labels_one_hot, learning_rate)
            
            epoch_loss += loss.item()
            batch_count += 1
            losses2.append(loss.item())
        
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.6f}")
    
    numpyGD_time = time.time() - numpyGD_start
    
    # train with sequential gradient descent
    print("\nTraining with numpySGD...")
    numpySGD_start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for images, labels in train_loader:
            # forward
            outputs = model3(images)
            # one-hot labels
            labels_one_hot = torch.zeros(labels.size(0), 10)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
            loss = criterion(outputs, labels_one_hot)
            
            # sequential backprop
            numpySGD(model3, images, labels_one_hot, learning_rate)
            
            epoch_loss += loss.item()
            batch_count += 1
            losses3.append(loss.item())
        
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.6f}")
    
    numpySGD_time = time.time() - numpySGD_start
    
    # test all 3 models
    print("\n" + "="*50)
    print("EVALUATION ON TEST SET")
    print("="*50)
    
    accuracies = []
    for i, model in enumerate([model1, model2, model3]):
        method_name = "normalGD" if i == 0 else "numpyGD" if i == 1 else "numpySGD"
        model.eval()
        correct = 0
        total = 0
        test_losses = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                # run model
                outputs = model(images)
                
                # calculate loss
                labels_one_hot = torch.zeros(labels.size(0), 10)
                labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
                test_loss = criterion(outputs, labels_one_hot)
                test_losses.append(test_loss.item())
                
                # check predictions
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct / total
        avg_test_loss = sum(test_losses) / len(test_losses)
        accuracies.append(test_accuracy)
        
        print(f"{method_name} - Test Accuracy: {test_accuracy:.2f}%")
        print(f"{method_name} - Average Test Loss: {avg_test_loss:.6f}")
        print(f"{method_name} - Correct Predictions: {correct}/{total}")
        
        # add timing info for each method
        if i == 0:
            print(f"{method_name} - Training Time: {normalGD_time:.2f} seconds")
        elif i == 1:
            print(f"{method_name} - Training Time: {numpyGD_time:.2f} seconds")
        else:
            print(f"{method_name} - Training Time: {numpySGD_time:.2f} seconds")
        print("-" * 30)
    
    print("="*50)
    
    # plot the losses
    plt.figure(figsize=(12, 8))
    line1, = plt.plot(losses1[50:], label='normalGD Loss', color='blue', markersize=1)
    line2, = plt.plot(losses2[50:], label='numpyGD Loss', color='red', markersize=1)
    line3, = plt.plot(losses3[50:], label='numpySGD Loss', color='green', markersize=1)
    mplcursors.cursor([line1, line2, line3], hover=True)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison: normalGD vs numpyGD vs numpySGD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    folderName = "single_layer_testing"
    os.makedirs(folderName, exist_ok=True)
    # check if file exists and add number suffix if needed
    base_filename = f'{folderName}/seed_{seed}_epochs_{epochs}_lr_{learning_rate}'
    filename = f'{base_filename}.png'
    counter = 1
    while os.path.exists(filename):
        filename = f'{base_filename}({counter}).png'
        counter += 1
    
    plt.savefig(filename)
    
    # report total time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

# run the comparison
if __name__ == "__main__":
    for i in range(1):
        compare_methods(seed=i, epochs=100, learning_rate=0.1)