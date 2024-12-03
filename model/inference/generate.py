import torch
from architecture.generator import Generator
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load the trained generator
generator = Generator(z_dim=100)
generator.load_state_dict(torch.load('model/checkpoints/generator.pth'))
generator.eval()

# Function to generate an image from a random noise vector (or text embedding in the future)
def generate_image(z_dim=100):
    z = torch.randn(1, z_dim)  # Random noise vector
    generated_image = generator(z).detach().numpy()
    generated_image = np.transpose(generated_image, (0, 2, 3, 1))  # Convert to HWC format

    # Convert to PIL Image and save
    generated_image = (generated_image[0] + 1) / 2.0  # Rescale to [0, 1]
    generated_image = np.uint8(generated_image * 255)
    image = Image.fromarray(generated_image)
    image.save('generated_image.png')
    image.show()

# Example usage
generate_image()
