import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 -> 784
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# Define simple NN
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3, (x1, x2)

# Training setup
loss_fn = nn.MSELoss()
model = SimpleNet()
learning_rate = 0.1

# Switch here between standard vs sequential
USE_SEQUENTIAL_GD = True

# Training loop
for epoch in range(10):
    for i, (inputs, labels) in enumerate(trainloader):
        outputs, (x1, x2) = model(inputs)
        
        # Convert labels to one-hot encoding for MSE loss
        one_hot_labels = torch.zeros(labels.size(0), 10)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
        
        loss = loss_fn(outputs, one_hot_labels)

        # Zero all gradients manually
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        if USE_SEQUENTIAL_GD:
            # Your sequential gradient descent
            loss.backward(retain_graph=True, inputs=(x2,))
            for param in model.fc3.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad

            grad_x2 = torch.autograd.grad(loss, x2, retain_graph=True)[0]
            x2.backward(grad_x2, retain_graph=True, inputs=(x1,))
            for param in model.fc2.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad

            grad_x1 = torch.autograd.grad(loss, x1)[0]
            x1.backward(grad_x1)
            for param in model.fc1.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad

        else:
            # Standard backprop
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad

        # Print every 100 batches
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}] Batch [{i}] Loss: {loss.item():.4f}")

print("\nDone training!")
