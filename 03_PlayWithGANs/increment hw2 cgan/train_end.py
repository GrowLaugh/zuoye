import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from cgan_end import Generator, Discriminator
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2   
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(model_gen, model_disc, dataloader, optimizer_disc, optimizer_gen, device, epoch, num_epochs):
    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        image_rgb = image_rgb.to(device) 
        image_semantic = image_semantic.to(device) 

        # Discriminator training
        noise = torch.randn_like(image_semantic).to(device)

        real_label = torch.ones(image_semantic.shape[0], 1).to(device)
        fake_label = torch.zeros(image_semantic.shape[0], 1).to(device)

        fake_images = model_gen(noise, image_semantic).detach()
        
        real_out = model_disc(image_rgb, image_semantic)
        fake_out = model_disc(fake_images, image_semantic) 
        
        optimizer_disc.zero_grad()

        loss_real_D = nn.BCELoss()(real_out, real_label)
        loss_fake_D = nn.BCELoss()(fake_out, fake_label)             
        loss_D = loss_fake_D + loss_real_D
        loss_D.backward()
        optimizer_disc.step()
        

        # Generator training
        noise = torch.randn_like(image_semantic).to(device)
        gen_images = model_gen(noise, image_semantic)                         
        out = model_disc(gen_images, image_semantic)  
        optimizer_gen.zero_grad()
        loss_G = nn.BCELoss()(out, real_label) + 0.25 * nn.MSELoss()(gen_images, image_rgb)
        loss_G.backward()
        optimizer_gen.step()

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_semantic, image_rgb, gen_images, 'train_results', epoch)
        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss_G: {loss_G.item():.6f}, Loss_D: {loss_D.item():.6f}')

def validate(model_gen, model_disc, dataloader, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.
    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    val_loss_d = 0.0
    val_loss_g = 0.0
    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)
            
            noise = torch.randn_like(image_semantic).to(device)
            real_label = torch.ones([image_semantic.shape[0], 1], dtype=torch.float32).to(device)
            fake_label = torch.zeros([image_semantic.shape[0], 1], dtype=torch.float32).to(device)
           
            fake_images = model_gen(noise, image_semantic).detach()
            real_out = model_disc(image_rgb, image_semantic)
            fake_out = model_disc(fake_images, image_semantic)

            loss_real_D = nn.BCELoss()(real_out, real_label)
            loss_fake_D = nn.BCELoss()(fake_out, fake_label)
            loss_D = loss_fake_D + loss_real_D
            val_loss_d += loss_D.item()

            noise = torch.randn_like(image_semantic).to(device)
            gen_images = model_gen(noise, image_semantic)   
            out = model_disc(gen_images, image_semantic)     
            loss_G = nn.BCELoss()(out, real_label)
            val_loss_g += loss_G.item()
            
            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_semantic, image_rgb, gen_images, 'val_results', epoch)

    # Calculate average validation loss
    avg_val_loss_g = val_loss_g / len(dataloader)
    avg_val_loss_d = val_loss_d / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Gen Validation Loss: {avg_val_loss_g:.4f}, Disc Validation Loss: {avg_val_loss_d:.4f}', len(dataloader))

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=4)
    
    # Initialize model, loss function, and optimizer
    # model = FullyConvNetwork().to(device)
    # criterion = nn.L1Loss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    generator = Generator(3, 3).to(device)
    discriminator = Discriminator(3).to(device)

    optimizer_generator = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    # scheduler = StepLR(optimizer, step_size=200, gamma=0.2)
    scheduler_gen = StepLR(optimizer_generator, step_size=100, gamma=0.6)
    scheduler_disc = StepLR(optimizer_discriminator, step_size=100, gamma=0.6)

    # Training loop
    num_epochs = 800
    for epoch in range(num_epochs):
        train_one_epoch(generator, discriminator, train_loader, optimizer_discriminator, optimizer_generator, device, epoch, num_epochs)
        validate(generator, discriminator, val_loader, device, epoch, num_epochs)
        
        # Step the scheduler after each epoch
        scheduler_gen.step()
        scheduler_disc.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            # torch.save(model.state_dict(), f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')
            torch.save(generator.state_dict(), f'checkpoints/gen_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/disc_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
