import torch
from torchvision.utils import save_image
import numpy as np
import os

from gan import Generator  


def load_generator(model_path, latent_dim):
    generator = Generator() 
    generator.load_state_dict(torch.load(model_path))
    generator.eval()
    return generator


def generate_images(generator, num_images, latent_dim, output_dir):
    with torch.no_grad():
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
    model_path = 'model/generator.pth' 
    latent_dim = 100 
    num_images = 10 
    output_dir = './output'  
    
    generator = load_generator(model_path, latent_dim)

    generate_images(generator, num_images, latent_dim, output_dir)
