import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import os

# --- 1. Hyperparameters and Setup ---
# Define the core parameters for the training process.
# These values are based on the original DCGAN paper and common practices.
batch_size = 128
image_size = 64  # We'll resize the images to 64x64
nc = 3           # Number of channels in the training images (3 for color images)
nz = 100         # Size of the latent z vector (i.e., size of generator input)
ngf = 64         # Size of feature maps in generator
ndf = 64         # Size of feature maps in discriminator
num_epochs = 25  # Number of training epochs
lr = 0.0002      # Learning rate for optimizers
beta1 = 0.5      # Beta1 hyperparameter for Adam optimizers

# Create a directory to save generated images and model checkpoints
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. Data Loading and Preprocessing ---
# We use torchvision to download and prepare the CelebA dataset.
# The transformations will resize, crop, and normalize the images.
# Normalizing to the range [-1, 1] is crucial as the generator uses tanh activation.

# NOTE: The first time you run this, it will download the CelebA dataset, which is large (over 1GB).
# This download process might take a considerable amount of time depending on your internet connection.
dataset = datasets.CelebA(
    root="./data",
    split='all',  # Use all available images
    download=True,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)

# Create the DataLoader to manage batches of data
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Decide which device to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 3. Model Architecture ---

# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# A. Generator Network
# The generator takes a random noise vector (z) and upsamples it through a series of
# transpose convolutional layers to create a 64x64 color image.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# B. Discriminator Network
# The discriminator takes a 64x64 image and downsamples it through convolutional layers,
# outputting a single probability indicating if the image is real or fake.
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# --- 4. Training Initialization ---
# Create instances of the models and move them to the GPU if available.
netG = Generator().to(device)
netD = Discriminator().to(device)

# Apply the weights_init function to randomly initialize all weights
netG.apply(weights_init)
netD.apply(weights_init)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create a batch of latent vectors that we will use to visualize
# the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# --- 5. The Training Loop ---
print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # ---------------------------------
        # (1) Update Discriminator network
        # ---------------------------------
        ## Train with all-real batch
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
        optimizerD.step()

        # ---------------------------------
        # (2) Update Generator network
        # ---------------------------------
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 100 == 0:
            print(
                f'[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}] '
                f'Loss_D: {errD.item():.4f} '
                f'Loss_G: {errG.item():.4f} '
                f'D(x): {D_x:.4f} '
                f'D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}'
            )

    # After each epoch, save a grid of generated images from our fixed_noise vector
    with torch.no_grad():
        fake_images = netG(fixed_noise).detach().cpu()
    save_image(
        fake_images,
        f"{output_dir}/fake_images_epoch_{epoch+1:02d}.png",
        normalize=True
    )

print("Training finished.")

# --- 6. Save the Trained Generator Model ---
generator_save_path = os.path.join(output_dir, "generator.pth")
torch.save(netG.state_dict(), generator_save_path)
print(f"Generator model saved to {generator_save_path}")


# --- 7. Testing the GAN (Inference) ---
# This section demonstrates how to load the trained generator and create new images.
# You can run this part independently after you have a saved 'generator.pth' file.
print("\n--- Starting GAN Testing/Inference ---")

# Path to the saved generator model
saved_model_path = os.path.join(output_dir, "generator.pth")

if os.path.exists(saved_model_path):
    # Instantiate a new generator
    test_generator = Generator().to(device)

    # Load the trained state dictionary
    test_generator.load_state_dict(torch.load(saved_model_path))

    # Set the model to evaluation mode
    test_generator.eval()

    # Generate a new batch of fake images
    with torch.no_grad():
        # Create a new batch of random noise
        test_noise = torch.randn(64, nz, 1, 1, device=device)
        # Generate images from the noise
        generated_images = test_generator(test_noise).detach().cpu()

    # Save the final generated images to a file
    final_output_path = os.path.join(output_dir, "final_generated_faces.png")
    save_image(generated_images, final_output_path, normalize=True)
    
    print(f"Successfully generated new faces and saved them to {final_output_path}")
    print("You can now open this file to see the results of your trained GAN.")

else:
    print(f"Could not find saved model at {saved_model_path}.")
    print("Please run the training script first to generate the model file.")

