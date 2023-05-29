import torch
import torch.nn as nn
import torch.optim as optim

# Define your neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set up your data
input_size = 10
hidden_size = 20
output_size = 5

# Create an instance of your neural network
model = NeuralNetwork(input_size, hidden_size, output_size)

# Define your loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generate some sample data
input_data = torch.randn(100, input_size)
target_data = torch.randn(100, output_size)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    output = model(input_data)
    loss = criterion(output, target_data)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every few epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Test the trained model
test_data = torch.randn(10, input_size)
with torch.no_grad():
    output = model(test_data)
    print("Output:")
    print(output)
