import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from gan import Generator, Discriminator
from preprocess_data import dataset

from generate_training_gif import store_progress_gif

print("Loading Data...")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Assuming generator, discriminator, optimizer_G, optimizer_D, criterion, dataloader are defined
latent_dim = 100
num_epochs = 50

print("Starting training process...")

generator = Generator()
discriminator = Discriminator()

print("Initialized generator and discriminator...")

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# Fixed noise for monitoring the progress
fixed_noise = torch.randn(64, latent_dim)

# Function to save a grid of generated images
def save_generated_images(generator, fixed_noise, epoch, output_dir="training_images"):
    with torch.no_grad():
        # Generate images from the fixed noise to monitor progress
        generated = generator(fixed_noise).detach().cpu()
    img_grid = vutils.make_grid(generated, padding=2, normalize=True)
    vutils.save_image(img_grid, f"{output_dir}/epoch_{epoch}.png")


for epoch in range(num_epochs):
    for i, (imgs,) in enumerate(dataloader):
        
        valid = torch.ones((imgs.size(0), 1), requires_grad=False)
        fake = torch.zeros((imgs.size(0), 1), requires_grad=False)

        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim)
        gen_imgs = generator(z)
        g_loss = criterion(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(imgs), valid)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # Logging training status
        if i % 50 == 0:  # Log every 50 batches
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} Loss D: {d_loss.item()}, loss G: {g_loss.item()}")

    if epoch % 1 == 0:  # Save images every epoch
        save_generated_images(generator, fixed_noise, epoch)    

print("Training process has finished.")

# Assuming generator and discriminator are your model instances
torch.save(generator.state_dict(), 'model/generator.pth')
torch.save(discriminator.state_dict(), 'model/discriminator.pth')
print("Saved Generator and Discriminator models.")

store_progress_gif()