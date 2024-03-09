import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim=100, img_shape=(1, 28, 20)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        
        # Calculate the initial size
        # The target is to match the final convolution output to img_shape dimensions
        # Let's start with an intermediate spatial size that will be upscaled to 28x20
        self.init_height, self.init_width = img_shape[1] // 4, img_shape[2] // 4
        self.l1_output_features = 128 * self.init_height * self.init_width
        self.l1 = nn.Sequential(nn.Linear(input_dim, self.l1_output_features))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # First upsampling to 14x10
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(size=(28, 20)),  # Upsample directly to the target size
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        # Reshape to the intermediate spatial size with appropriate depth
        out = out.view(-1, 128, self.init_height, self.init_width)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 20)):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, 3, stride=2, padding=0),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(256 * 3 * 2, 1),  # Corrected to match actual output size
            nn.Sigmoid()
        )

    def forward(self, img):
        # print("In discriminator -----")
        # print(img.shape)
        validity = self.model(img)
        return validity

