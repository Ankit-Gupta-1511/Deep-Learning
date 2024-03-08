import torch
from torchvision.utils import save_image
import numpy as np
import os

# Import your model's architecture
from gan import Generator  # Adjust this import path as needed


def load_generator(model_path, latent_dim):
    generator = Generator()  # Initialize generator with the same latent_dim used in training
    generator.load_state_dict(torch.load(model_path))
    generator.eval()  # Set to evaluation mode
    return generator


def generate_images(generator, num_images, latent_dim, output_dir):
    with torch.no_grad():  # No need to track gradients
        # Generate random latent vectors
        z = torch.randn(num_images, latent_dim)
        # Generate images
        images = generator(z)
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Save images
        for i, image in enumerate(images):
            save_image(image, os.path.join(output_dir, f"image_{i+1}.png"))
    print(f"Generated {num_images} images in {output_dir}")



if __name__ == "__main__":
    # Configuration
    model_path = 'model/generator.pth'  # Path to the saved model
    latent_dim = 100  # Ensure this matches the training configuration
    num_images = 10  # Number of images to generate
    output_dir = './output'  # Directory to save generated images
    
    # Load the generator
    generator = load_generator(model_path, latent_dim)
    
    # Generate images
    generate_images(generator, num_images, latent_dim, output_dir)
