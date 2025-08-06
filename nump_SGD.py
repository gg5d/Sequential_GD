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

# train_dataset = torch.utils.data.Subset(train_dataset, range(5000))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# 2. Define a very simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 16)  # first layer
        self.fc2 = nn.Linear(16, 10)     # output layer (10 classes)

    def forward(self, x):
        x = x.view(-1, 28*28)        # flatten 28x28 to 784
        x = torch.sigmoid(self.fc1(x))  # activation (better for gradient flow)
        x = torch.sigmoid(self.fc2(x))  # activation
        return x

def create_model_with_seed(seed):
    """Create a model with fixed random seed for reproducible results"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return SimpleNN()

def standardGD(model, loss, learning_rate):
    # Backward pass
    loss.backward()
    
    # Efficient gradient descent with vectorized operations
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                # Use in-place operations for better memory efficiency
                param.data.add_(-learning_rate * param.grad)
                param.grad.zero_()

def numpyGD(model, images, labels_one_hot, learning_rate):
    #Manual backprop + gradient descent in NumPy for 2-layer NN with sigmoid + MSE.
    
    # Convert tensors to numpy (make sure they are on CPU)
    x = images.view(-1, 28*28).numpy()           # (batch, 784)
    y = labels_one_hot.numpy()                   # (batch, 10)

    # Extract weights/biases
    W1 = model.fc1.weight.data.numpy()           # (16, 784)
    b1 = model.fc1.bias.data.numpy()             # (16,)
    W2 = model.fc2.weight.data.numpy()           # (10, 16)
    b2 = model.fc2.bias.data.numpy()             # (10,)

    # --- Forward pass (NumPy) ---
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Layer 1
    z1 = x @ W1.T + b1          # (batch, 16)
    a1 = sigmoid(z1)            # (batch, 16)
    # Layer 2
    z2 = a1 @ W2.T + b2         # (batch, 10)
    a2 = sigmoid(z2)            # (batch, 10)
    
    # --- Compute loss ---
    loss = np.mean((a2 - y) ** 2)

    # --- Backprop (manual chain rule) ---
    # dL/da2 = 2*(a2 - y)/batch
    batch_size = x.shape[0]
    dA2 = 2 * (a2 - y) / batch_size

    # sigmoid'(z) = a * (1 - a)
    dZ2 = dA2 * a2 * (1 - a2)          # (batch, 10)

    # Gradients for W2 and b2
    dW2 = dZ2.T @ a1                   # (10, 16)
    db2 = np.sum(dZ2, axis=0)          # (10,)

    # Propagate to layer 1
    dA1 = dZ2 @ W2                     # (batch, 16)
    dZ1 = dA1 * a1 * (1 - a1)          # (batch, 16)

    # Gradients for W1 and b1
    dW1 = dZ1.T @ x                    # (16, 784)
    db1 = np.sum(dZ1, axis=0)          # (16,)

    # --- Gradient Descent Update ---
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # --- Copy updated params back to model ---
    model.fc1.weight.data.copy_(torch.from_numpy(W1))
    model.fc1.bias.data.copy_(torch.from_numpy(b1))
    model.fc2.weight.data.copy_(torch.from_numpy(W2))
    model.fc2.bias.data.copy_(torch.from_numpy(b2))

    return loss



def numpySGD(model, images, labels_one_hot, learning_rate):
    #SEQUENTIALGD (SGD)
    #Manual backprop + gradient descent in NumPy for 2-layer NN with sigmoid + MSE.
    
    # Convert tensors to numpy (make sure they are on CPU)
    x = images.view(-1, 28*28).numpy()           # (batch, 784)
    y = labels_one_hot.numpy()                   # (batch, 10)

    # Extract weights/biases
    W1 = model.fc1.weight.data.numpy()           # (16, 784)
    b1 = model.fc1.bias.data.numpy()             # (16,)
    W2 = model.fc2.weight.data.numpy()           # (10, 16)
    b2 = model.fc2.bias.data.numpy()             # (10,)

    # --- Forward pass (NumPy) ---
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Layer 1
    z1 = x @ W1.T + b1          # (batch, 16)
    a1 = sigmoid(z1)            # (batch, 16)
    # Layer 2
    z2 = a1 @ W2.T + b2         # (batch, 10)
    a2 = sigmoid(z2)            # (batch, 10)
    
    # --- Compute loss ---
    loss = np.mean((a2 - y) ** 2)

    # --- Backprop (manual chain rule) ---
    # dL/da2 = 2*(a2 - y)/batch
    batch_size = x.shape[0]
    dA2 = 2 * (a2 - y) / batch_size

    # sigmoid'(z) = a * (1 - a)
    dZ2 = dA2 * a2 * (1 - a2)          # (batch, 10)

    # Gradients for W2 and b2
    dW2 = dZ2.T @ a1                   # (10, 16)
    db2 = np.sum(dZ2, axis=0)          # (10,)

    # --- Gradient Descent Update ---
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Propagate to layer 1
    dA1 = dZ2 @ W2                     # (batch, 16)
    dZ1 = dA1 * a1 * (1 - a1)          # (batch, 16)

    # Gradients for W1 and b1
    dW1 = dZ1.T @ x                    # (16, 784)
    db1 = np.sum(dZ1, axis=0)          # (16,)

    # --- Gradient Descent Update ---
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # --- Copy updated params back to model ---
    model.fc1.weight.data.copy_(torch.from_numpy(W1))
    model.fc1.bias.data.copy_(torch.from_numpy(b1))
    model.fc2.weight.data.copy_(torch.from_numpy(W2))
    model.fc2.bias.data.copy_(torch.from_numpy(b2))

    return loss

def compare_methods(seed=42, epochs=100, learning_rate=0.01):
    """Compare numpyGD and numpySGD methods with same starting weights"""
    
    # Create two models with same seed
    model1 = create_model_with_seed(seed)
    model2 = create_model_with_seed(seed)
    
    criterion = nn.MSELoss()
    losses1 = []
    losses2 = []
    
    print(f"Comparing methods with seed {seed}")
    print("="*50)
    
    # Train model1 with numpyGD
    print("Training with numpyGD...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for images, labels in train_loader:
            # Forward pass
            outputs = model1(images)
            # Convert labels to one-hot encoding for MSE
            labels_one_hot = torch.zeros(labels.size(0), 10)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
            loss = criterion(outputs, labels_one_hot)
            
            # Backward pass and gradient descent
            numpyGD(model1, images, labels_one_hot, learning_rate)
            
            epoch_loss += loss.item()
            batch_count += 1
            losses1.append(loss.item())
        
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.6f}")
    
    # Train model2 with numpySGD
    print("\nTraining with numpySGD...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for images, labels in train_loader:
            # Forward pass
            outputs = model2(images)
            # Convert labels to one-hot encoding for MSE
            labels_one_hot = torch.zeros(labels.size(0), 10)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
            loss = criterion(outputs, labels_one_hot)
            
            # Backward pass and gradient descent
            numpySGD(model2, images, labels_one_hot, learning_rate)
            
            epoch_loss += loss.item()
            batch_count += 1
            losses2.append(loss.item())
        
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.6f}")
    
    # Evaluate both models
    print("\n" + "="*50)
    print("EVALUATION ON TEST SET")
    print("="*50)
    
    accuracies = []
    for i, model in enumerate([model1, model2]):
        method_name = "numpyGD" if i == 0 else "numpySGD"
        model.eval()
        correct = 0
        total = 0
        test_losses = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                # Forward pass
                outputs = model(images)
                
                # Convert labels to one-hot for loss calculation
                labels_one_hot = torch.zeros(labels.size(0), 10)
                labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
                test_loss = criterion(outputs, labels_one_hot)
                test_losses.append(test_loss.item())
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct / total
        avg_test_loss = sum(test_losses) / len(test_losses)
        accuracies.append(test_accuracy)
        
        print(f"{method_name} - Test Accuracy: {test_accuracy:.2f}%")
        print(f"{method_name} - Average Test Loss: {avg_test_loss:.6f}")
        print(f"{method_name} - Correct Predictions: {correct}/{total}")
        print("-" * 30)
    
    print("="*50)
    
    # Plot both losses
    plt.figure(figsize=(12, 8))
    line1, = plt.plot(losses1[50:], label='numpyGD Loss', color='blue', markersize=1)
    line2, = plt.plot(losses2[50:], label='numpySGD Loss', color='red', markersize=1)
    mplcursors.cursor([line1, line2], hover=True)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison: numpyGD vs numpySGD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 4. Training loop - now using comparison function
if __name__ == "__main__":
    compare_methods(seed=42, epochs=100, learning_rate=0.01)


