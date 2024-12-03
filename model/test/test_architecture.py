import torch
from architecture.generator import Generator
from architecture.discriminator import Discriminator

def test_generator():
    generator = Generator(z_dim=100)
    z = torch.randn(1, 100)  # Single random noise vector
    generated_image = generator(z)
    assert generated_image.shape == (1, 3, 28, 28), "Generator output shape mismatch"

def test_discriminator():
    discriminator = Discriminator()
    fake_img = torch.randn(1, 3, 28, 28)  # Fake image
    validity = discriminator(fake_img)
    assert validity.shape == (1, 1), "Discriminator output shape mismatch"
    assert validity.item() >= 0 and validity.item() <= 1, "Discriminator output range error"

test_generator()
test_discriminator()
print("Architecture tests passed!")
