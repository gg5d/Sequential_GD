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

train_dataset = torch.utils.data.Subset(train_dataset, range(500))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

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

model = SimpleNN()

# 3. Define loss and learning rate
criterion = nn.MSELoss()
learning_rate = 0.01

def standardGD(model, loss, learning_rate):
    """Perform backward pass and manual gradient descent"""
    # Backward pass
    loss.backward()
    
    # Manual gradient descent (vanilla SGD)
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
            param.grad.zero_()

def numpyGD(model, loss, learning_rate):
    """Perform backward pass and gradient descent using pure numpy"""
    # Get model parameters as numpy arrays
    fc1_weight = model.fc1.weight.data.numpy()
    fc1_bias = model.fc1.bias.data.numpy()
    fc2_weight = model.fc2.weight.data.numpy()
    fc2_bias = model.fc2.bias.data.numpy()
    
    # Convert input data to numpy
    images_np = images.numpy()
    labels_one_hot_np = labels_one_hot.numpy()
    
    epsilon = 1e-4
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def forward_numpy(x, w1, b1, w2, b2):
        """Pure numpy forward pass"""
        # Flatten input
        x = x.reshape(-1, 784)
        
        # First layer
        z1 = np.dot(x, w1.T) + b1
        a1 = sigmoid(z1)
        
        # Second layer
        z2 = np.dot(a1, w2.T) + b2
        a2 = sigmoid(z2)
        
        return a2
    
    def mse_loss_numpy(predictions, targets):
        """Pure numpy MSE loss"""
        return np.mean((predictions - targets) ** 2)
    
    # Compute current loss with numpy
    current_outputs = forward_numpy(images_np, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
    current_loss = mse_loss_numpy(current_outputs, labels_one_hot_np)
    print(f"Starting numpyGD with loss: {current_loss:.6f}")
    
    print("Computing gradients for all parameters...")
    
    # fc1 weights
    fc1_weight_grad = np.zeros_like(fc1_weight)
    for i in range(fc1_weight.shape[0]):
        for j in range(fc1_weight.shape[1]):
            original_weight = fc1_weight[i, j]
            fc1_weight[i, j] += epsilon
            
            outputs_plus = forward_numpy(images_np, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
            loss_plus = mse_loss_numpy(outputs_plus, labels_one_hot_np)
            
            fc1_weight[i, j] = original_weight
            fc1_weight_grad[i, j] = (loss_plus - current_loss) / epsilon
    
    # fc1 bias
    fc1_bias_grad = np.zeros_like(fc1_bias)
    for i in range(fc1_bias.shape[0]):
        original_bias = fc1_bias[i]
        fc1_bias[i] += epsilon
        
        outputs_plus = forward_numpy(images_np, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
        loss_plus = mse_loss_numpy(outputs_plus, labels_one_hot_np)
        
        fc1_bias[i] = original_bias
        fc1_bias_grad[i] = (loss_plus - current_loss) / epsilon
    
    # fc2 weights
    fc2_weight_grad = np.zeros_like(fc2_weight)
    for i in range(fc2_weight.shape[0]):
        for j in range(fc2_weight.shape[1]):
            original_weight = fc2_weight[i, j]
            fc2_weight[i, j] += epsilon
            
            outputs_plus = forward_numpy(images_np, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
            loss_plus = mse_loss_numpy(outputs_plus, labels_one_hot_np)
            
            fc2_weight[i, j] = original_weight
            fc2_weight_grad[i, j] = (loss_plus - current_loss) / epsilon
    
    # fc2 bias
    fc2_bias_grad = np.zeros_like(fc2_bias)
    for i in range(fc2_bias.shape[0]):
        original_bias = fc2_bias[i]
        fc2_bias[i] += epsilon
        
        outputs_plus = forward_numpy(images_np, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
        loss_plus = mse_loss_numpy(outputs_plus, labels_one_hot_np)
        
        fc2_bias[i] = original_bias
        fc2_bias_grad[i] = (loss_plus - current_loss) / epsilon
    
    print(f"Gradient computation complete.")
    print(f"fc1_weight grad max: {np.max(fc1_weight_grad):.6f}")
    print(f"fc1_bias grad max: {np.max(fc1_bias_grad):.6f}")
    print(f"fc2_weight grad max: {np.max(fc2_weight_grad):.6f}")
    print(f"fc2_bias grad max: {np.max(fc2_bias_grad):.6f}")
    
    # Update ALL parameters using gradient descent
    fc1_weight -= learning_rate * fc1_weight_grad
    fc1_bias -= learning_rate * fc1_bias_grad
    fc2_weight -= learning_rate * fc2_weight_grad
    fc2_bias -= learning_rate * fc2_bias_grad
    
    # Update model parameters
    model.fc1.weight.data = torch.tensor(fc1_weight)
    model.fc1.bias.data = torch.tensor(fc1_bias)
    model.fc2.weight.data = torch.tensor(fc2_weight)
    model.fc2.bias.data = torch.tensor(fc2_bias)
    
    print(f"Updated all parameters. New loss will be computed in next iteration.")


# 4. Training loop
losses = []
for epoch in range(10):  # just 3 epochs for simplicity
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        # Convert labels to one-hot encoding for MSE
        labels_one_hot = torch.zeros(labels.size(0), 10)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
        loss = criterion(outputs, labels_one_hot)
        
        # Backward pass and gradient descent
        numpyGD(model, loss, learning_rate)
        
        
        losses.append(loss.item())
    print(f"Epoch {epoch+1} completed")



# 5. Plot loss graph
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


