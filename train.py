import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os
import numpy as np
from torchviz import make_dot

from gan import Generator, Discriminator
from preprocess_data import dataset

from generate_training_gif import store_progress_gif

print("Loading Data...")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

latent_dim = 100
num_epochs = 50
img_shape = (1, 28, 20)  # actual image size

print("Starting training process...")

generator = Generator(input_dim=latent_dim, img_shape=img_shape)
discriminator = Discriminator(img_shape=img_shape)

print("Initialized generator and discriminator...")

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

generator_output = None
discriminator_output = None

if not os.path.exists('training_output'):
    os.makedirs('training_output')


def save_generated_images(generator_output, epoch, output_dir="training_images"):
    with torch.no_grad():
        fixed_gen_imgs = generator_output.detach().cpu()
    img_grid = vutils.make_grid(fixed_gen_imgs, padding=2, normalize=True)
    vutils.save_image(img_grid, f"{output_dir}/epoch_{epoch}.png")

for epoch in range(num_epochs):
    for i, (imgs,) in enumerate(dataloader):
        
        valid = torch.ones((imgs.size(0), 1), device=imgs.device)
        fake = torch.zeros((imgs.size(0), 1), device=imgs.device)

        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()

        z = torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))  # Generate random noise
        gen_imgs = generator(z)
        
        generator_output = gen_imgs
        discriminator_output = discriminator(gen_imgs)
        g_loss = criterion(discriminator_output, valid)
        
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(imgs.unsqueeze(1)), valid)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        optimizer_D.step()
        
        if i % 50 == 0:  # Log every 50 batches
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} Loss D: {d_loss.item()}, loss G: {g_loss.item()}")

    # Generate and save images at the end of each epoch
    save_generated_images(generator_output, epoch)    

print("Training process has finished.")

# Save models
torch.save(generator.state_dict(), 'model/generator.pth')
torch.save(discriminator.state_dict(), 'model/discriminator.pth')
print("Saved Generator and Discriminator models.")

store_progress_gif()

gen_vis_graph = make_dot(generator_output, params=dict(generator.named_parameters()))
gen_vis_graph.render('training_output/generator_architecture', format='png')

# Visualize the Discriminator
disc_vis_graph = make_dot(discriminator_output, params=dict(discriminator.named_parameters()))
disc_vis_graph.render('training_output/discriminator_architecture', format='png')