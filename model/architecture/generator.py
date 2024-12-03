import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28 * 28 * 3)  # Example for 28x28 RGB images

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x.view(x.size(0), 3, 28, 28)  # Reshaping to image dimensions

# Example usage:
# generator = Generator(z_dim=100)
# noise = torch.randn(16, 100)  # Batch of 16 random noise vectors
# generated_images = generator(noise)
