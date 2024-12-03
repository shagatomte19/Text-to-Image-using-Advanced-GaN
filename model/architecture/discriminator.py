import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, img):
        x = img.view(img.size(0), -1)  # Flatten the image
        x = torch.leaky_relu(self.fc1(x), 0.2)
        x = torch.leaky_relu(self.fc2(x), 0.2)
        x = torch.leaky_relu(self.fc3(x), 0.2)
        return torch.sigmoid(self.fc4(x))

# Example usage:
# discriminator = Discriminator()
# real_img = torch.randn(16, 3, 28, 28)  # Batch of 16 real images
# validity = discriminator(real_img)  # Output between 0 and 1
