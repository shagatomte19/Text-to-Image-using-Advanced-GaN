import torch
import torch.optim as optim
from architecture.generator import Generator
from architecture.discriminator import Discriminator
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import os

# Initialize the generator and discriminator
z_dim = 100  # Size of the random noise vector
generator = Generator(z_dim)
discriminator = Discriminator()

# Optimizers
lr = 0.0002
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):  # Assuming train_loader is defined elsewhere
        batch_size = imgs.size(0)
        valid = torch.ones(batch_size, 1)  # Real labels
        fake = torch.zeros(batch_size, 1)  # Fake labels

        # -----------------
        # Train Discriminator
        # -----------------
        optimizer_D.zero_grad()

        # Real images
        real_imgs = imgs
        real_loss = adversarial_loss(discriminator(real_imgs), valid)

        # Fake images
        z = torch.randn(batch_size, z_dim)  # Generate random noise
        fake_imgs = generator(z)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)

        # Total loss for discriminator
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generate fake images and calculate the loss
        g_loss = adversarial_loss(discriminator(fake_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch}/{epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")
