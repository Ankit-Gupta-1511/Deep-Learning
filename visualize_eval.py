import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

from gan import Generator  # Adjust this import path as needed


def load_generator(model_path, latent_dim):
    generator = Generator()  # Initialize generator with the same latent_dim used in training
    generator.load_state_dict(torch.load(model_path))
    generator.eval()  # Set to evaluation mode
    return generator


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