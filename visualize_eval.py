import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

from gan import Generator


def load_generator(model_path, latent_dim):
    generator = Generator() 
    generator.load_state_dict(torch.load(model_path))
    generator.eval() 
    return generator


if __name__ == "__main__":
    # Configuration
    model_path = 'model/generator.pth' 
    latent_dim = 100 
    num_images = 10 
    output_dir = './output' 
    
    generator = load_generator(model_path, latent_dim)
    
    generate_images(generator, num_images, latent_dim, output_dir)