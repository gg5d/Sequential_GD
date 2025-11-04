import mplcursors
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time


# flexible MLP that follows layer_sizes
class FlexibleNN(nn.Module):
    def __init__(self, layer_sizes, activation='sigmoid'):
        super(FlexibleNN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
        self.activation = activation

    def forward(self, x):
        x = x.view(-1, layer_sizes[0])  # flatten to match input size
        for i, lin in enumerate(self.layers):
            if i < len(self.layers) - 1:  # hidden layers
                x = self._apply_activation(lin(x))
            else:  # final layer (no activation)
                x = lin(x)
        return x

    def _apply_activation(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'leaky_relu':
            return torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        else:
            return torch.sigmoid(x)


def create_model_with_seed(seed, activation='sigmoid'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return FlexibleNN(layer_sizes, activation=activation)


def standardGD(model, loss, learning_rate):
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data.add_(-learning_rate * param.grad)
                param.grad.zero_()


# helper: forward pass that caches activations for manual backprop
def _forward_cache(x, Ws, bs, activation='sigmoid'):
    # x: (batch, in_dim); Ws: [W1..WL]; bs: [b1..bL]
    activations = [x]
    a = x
    for W, b in zip(Ws, bs):
        z = a @ W.T + b
        if activation == 'relu':
            a = torch.relu(z)
        elif activation == 'leaky_relu':
            a = torch.nn.functional.leaky_relu(z, negative_slope=0.01)
        else:
            a = 1 / (1 + torch.exp(-z))    # sigmoid
        activations.append(a)
    return activations


def _activation_derivative(a, activation):
    if activation == 'relu':
        return (a > 0).float()
    elif activation == 'leaky_relu':
        grad = torch.ones_like(a)
        grad[a < 0] = 0.01
        return grad
    else:
        return a * (1 - a)  # sigmoid'


def pytorchGD(model, images, labels_one_hot, learning_rate, activation='sigmoid'):
    # STANDARD GD - updates both layers at the same time
    # same backprop setup as pytorchSGD but different update order
    
    # convert to pytorch
    x = images.view(-1, layer_sizes[0])           # (batch, in_dim)
    y = labels_one_hot                            # (batch, out_dim)

    # pull out weights/biases (clones so we can compute/assign)
    Ws = [lin.weight.data.clone() for lin in model.layers]
    bs = [lin.bias.data.clone()   for lin in model.layers]

    # forward pass
    activations = _forward_cache(x, Ws, bs, activation=activation)
    aL = activations[-1]
    
    # compute loss
    loss = torch.mean((aL - y) ** 2)

    # backprop (chain rule)
    batch_size = x.shape[0]
    out_dim = y.shape[1]
    dA = 2 * (aL - y) / (batch_size * out_dim)

    # grads per layer
    dWs = [None] * len(Ws)
    dbs = [None] * len(bs)

    # go from last to first
    for l in reversed(range(len(Ws))):
        a_prev = activations[l]
        a_cur  = activations[l+1]
        dZ = dA * _activation_derivative(a_cur, activation)
        dWs[l] = dZ.T @ a_prev
        dbs[l] = torch.sum(dZ, dim=0)
        dA = dZ @ Ws[l]

    # UPDATE BOTH LAYERS TOGETHER (this is the standard way)
    for l, lin in enumerate(model.layers):
        Ws[l] -= learning_rate * dWs[l]
        bs[l] -= learning_rate * dbs[l]
        lin.weight.data.copy_(Ws[l])
        lin.bias.data.copy_(bs[l])

    return loss.item()


def pytorchSGD(model, images, labels_one_hot, learning_rate, activation='sigmoid'):
    # SEQUENTIAL GD - updates layer 2 first then layer 1
    # same backprop setup as pytorchGD but different update order
    
    # convert to pytorch
    x = images.view(-1, layer_sizes[0])           # (batch, in_dim)
    y = labels_one_hot                            # (batch, out_dim)

    # clones so we can update per-layer in sequence
    Ws = [lin.weight.data.clone() for lin in model.layers]
    bs = [lin.bias.data.clone()   for lin in model.layers]

    # forward pass
    activations = _forward_cache(x, Ws, bs, activation=activation)
    aL = activations[-1]
    
    # compute loss
    loss = torch.mean((aL - y) ** 2)

    # backprop (chain rule)
    batch_size = x.shape[0]
    out_dim = y.shape[1]
    dA = 2 * (aL - y) / (batch_size * out_dim)

    # go from last to first, updating each layer as we go
    for l in reversed(range(len(Ws))):
        a_prev = activations[l]
        a_cur  = activations[l+1]
        dZ = dA * _activation_derivative(a_cur, activation)
        dW = dZ.T @ a_prev
        db = torch.sum(dZ, dim=0)

        # UPDATE LAYER l FIRST (this is the sequential part)
        Ws[l] -= learning_rate * dW
        bs[l] -= learning_rate * db

        # now backprop to previous using UPDATED weights
        dA = dZ @ Ws[l]

    # copy everything back
    for l, lin in enumerate(model.layers):
        lin.weight.data.copy_(Ws[l])
        lin.bias.data.copy_(bs[l])

    return loss.item()


def compare_methods(seed=42, epochs=100, learning_rate=0.01, activation='sigmoid', device='cpu'):
    start_time = time.time()
    model1 = create_model_with_seed(seed, activation)
    model2 = create_model_with_seed(seed, activation)
    model3 = create_model_with_seed(seed, activation)

    # move models to device
    model1.to(device)
    model2.to(device)
    model3.to(device)
    
    criterion = nn.MSELoss()
    losses1, losses2, losses3 = [], [], []
    
    print(f"Comparing methods with seed {seed}, activation={activation}")
    print("="*50)
    
    print("Training with normalGD...")
    normalGD_start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model1(images)
            labels_one_hot = torch.zeros(labels.size(0), layer_sizes[-1], device=device)
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
            images, labels = images.to(device), labels.to(device)
            outputs = model2(images)
            labels_one_hot = torch.zeros(labels.size(0), layer_sizes[-1], device=device)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
            loss_val = pytorchGD(model2, images, labels_one_hot, learning_rate, activation)
            epoch_loss += loss_val
            batch_count += 1
            losses2.append(loss_val)
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.6f}")
    pytorchGD_time = time.time() - pytorchGD_start
    
    print("\nTraining with pytorchSGD...")
    pytorchSGD_start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model3(images)
            labels_one_hot = torch.zeros(labels.size(0), layer_sizes[-1], device=device)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
            loss_val = pytorchSGD(model3, images, labels_one_hot, learning_rate, activation)
            epoch_loss += loss_val
            batch_count += 1
            losses3.append(loss_val)
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
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                labels_one_hot = torch.zeros(labels.size(0), layer_sizes[-1], device=device)
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
    plt.title(f'Training Loss Comparison: normalGD vs pytorchGD vs pytorchSGD ({activation})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    folderName = "5_layer_classification_testing"
    os.makedirs(folderName, exist_ok=True)
    base_filename = f'{folderName}/seed_{seed}_epochs_{epochs}_lr_{learning_rate}_{activation}'
    filename = f'{base_filename}.png'
    counter = 1
    while os.path.exists(filename):
        filename = f'{base_filename}({counter}).png'
        counter += 1
    plt.savefig(filename)

    torch.save(model1.state_dict(), "temp_model.pt")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

    return model1



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ======= Synthetic 2D Data =======
    n_per_class = 10
    class0 = np.array([
        [0.1, 0.1], [0.25, 0.4], [0.1, 0.5], [0.6, 0.9],
        [0.4, 0.2], [0.25, 0.6], [0.8, 0.1], [0.1, 0.8],
        [0.7, 0.2], [0.5, 0.9]
    ])
    class1 = np.array([
        [0.6, 0.35], [0.5, 0.6], [0.8, 0.4], [0.4, 0.4],
        [0.7, 0.6], [0.35, 0.9], [0.3, 0.8], [0.8, 0.8],
        [0.4, 0.6], [0.8, 0.9]
    ])

    X = np.vstack([class0, class1]).astype(np.float32)
    y = np.array([0]*n_per_class + [1]*n_per_class)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False)

    # Architecture: input=2 -> hidden -> output=2
    layer_sizes = [2, 16, 16, 2]

    # ======= Train and get model =======
    trained_model = compare_methods(seed=10, epochs=500000, learning_rate=0.005, activation='relu')

    # ======= Visualization =======
    trained_model.eval()
    xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = trained_model(grid)
        Z = torch.argmax(Z, axis=1).numpy().reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.viridis, alpha=0.8)
    plt.scatter(class0[:, 0], class0[:, 1], c='cyan', marker='^', edgecolor='k', s=80)
    plt.scatter(class1[:, 0], class1[:, 1], c='orange', edgecolor='k', s=80)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Decision Boundary Learned by MLP")
    plt.show()

