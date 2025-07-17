import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import copy
import matplotlib.pyplot as plt
import mplcursors

# --------------------
# Load MNIST dataset
# --------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 -> 784
])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainset = torch.utils.data.Subset(trainset, range(500))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=12, shuffle=True)

# --------------------
# Build a 5-layer network
# --------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 24)
        self.fc3 = nn.Linear(24, 18)
        self.fc4 = nn.Linear(18, 12)
        self.fc5 = nn.Linear(12, 10)
    def forward(self, x):
        x1 = torch.sigmoid(self.fc1(x))
        x2 = torch.sigmoid(self.fc2(x1))
        x3 = torch.sigmoid(self.fc3(x2))
        x4 = torch.sigmoid(self.fc4(x3))
        out = self.fc5(x4)
        return out, (x1, x2, x3, x4)

# --------------------
# Setup
# --------------------
learning_rate = 0.01
loss_fn = nn.MSELoss()
batch_limit = 200

standard_losses = []
sequential_losses = []

# --------------------
# 1. Standard Backprop
# --------------------
model = SimpleNet()
initial_state = copy.deepcopy(model.state_dict())

model.load_state_dict(initial_state)
print("\n--- Standard Backpropagation ---")
for epoch in range(10):
    for i, (inputs, labels) in enumerate(trainloader):
        if i >= batch_limit: break
        outputs, _ = model(inputs)
        one_hot_labels = torch.zeros(labels.size(0), 10)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        loss = loss_fn(outputs, one_hot_labels)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.data -= learning_rate * p.grad

        standard_losses.append(loss.item())

# --------------------
# 2. Progressive Sequential GD (single forward, 5-layer)
# --------------------
model.load_state_dict(initial_state)
print("\n--- Progressive Sequential GD (5-layer) ---")

for epoch in range(10):
    for i, (inputs, labels) in enumerate(trainloader):
        if i >= batch_limit: break

        one_hot_labels = torch.zeros(labels.size(0), 10)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

        inputs.requires_grad = False

        # --- Forward pass once ---
        x1 = torch.sigmoid(model.fc1(inputs))
        x2 = torch.sigmoid(model.fc2(x1))
        x3 = torch.sigmoid(model.fc3(x2))
        x4 = torch.sigmoid(model.fc4(x3))
        outputs = model.fc5(x4)
        loss = loss_fn(outputs, one_hot_labels)

        # --- Backprop for fc5 ---
        grads_fc5 = torch.autograd.grad(loss, model.fc5.parameters(), retain_graph=True)
        for p, g in zip(model.fc5.parameters(), grads_fc5):
            p.data -= learning_rate * g

        # --- Backprop for fc4 ---
        dL_dx4 = torch.autograd.grad(loss, x4, retain_graph=True)[0]
        grads_fc4 = torch.autograd.grad(x4, model.fc4.parameters(), grad_outputs=dL_dx4, retain_graph=True)
        for p, g in zip(model.fc4.parameters(), grads_fc4):
            p.data -= learning_rate * g

        # --- Backprop for fc3 ---
        dL_dx3 = torch.autograd.grad(x4, x3, grad_outputs=dL_dx4, retain_graph=True)[0]
        grads_fc3 = torch.autograd.grad(x3, model.fc3.parameters(), grad_outputs=dL_dx3, retain_graph=True)
        for p, g in zip(model.fc3.parameters(), grads_fc3):
            p.data -= learning_rate * g

        # --- Backprop for fc2 ---
        dL_dx2 = torch.autograd.grad(x3, x2, grad_outputs=dL_dx3, retain_graph=True)[0]
        grads_fc2 = torch.autograd.grad(x2, model.fc2.parameters(), grad_outputs=dL_dx2, retain_graph=True)
        for p, g in zip(model.fc2.parameters(), grads_fc2):
            p.data -= learning_rate * g

        # --- Backprop for fc1 ---
        dL_dx1 = torch.autograd.grad(x2, x1, grad_outputs=dL_dx2)[0]
        grads_fc1 = torch.autograd.grad(x1, model.fc1.parameters(), grad_outputs=dL_dx1)
        for p, g in zip(model.fc1.parameters(), grads_fc1):
            p.data -= learning_rate * g

        sequential_losses.append(loss.item())

# --------------------
# Plot
# --------------------
plt.figure(figsize=(12,6))
line1, = plt.plot(standard_losses, label='Standard Backprop')
line2, = plt.plot(sequential_losses, label='Progressive Sequential GD (5-layer)')
plt.xlabel('Batch (across all epochs)')
plt.ylabel('Loss')
plt.title('Loss Comparison: Standard vs Progressive Sequential')
plt.legend()
plt.grid(True)
mplcursors.cursor([line1, line2], hover=True)
plt.show()
