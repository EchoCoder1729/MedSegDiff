import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class NumpyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory containing "images" and "masks" subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = os.path.join(root_dir, "saved_images")
        self.mask_dir = os.path.join(root_dir, "saved_masks")

        self.name_list = sorted(glob(os.path.join(self.image_dir, "*.npy")))
        self.label_list = sorted(glob(os.path.join(self.mask_dir, "*.npy")))
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        # Load image and mask from .npy files
        img_path = self.name_list[idx]
        msk_path = self.label_list[idx]

        # Load .npy files
        image = np.load(img_path)
        mask = np.load(msk_path)

        # Convert numpy arrays to torch tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        # Ensure the image has 3 channels and mask has 1 channel
        if image.ndim == 2:  # If the image is grayscale
            image = image.unsqueeze(0)  # Add a channel dimension
        elif image.ndim == 3 and image.shape[0] != 3:  # If the channel is not the first dimension
            image = image.permute(2, 0, 1)  # Reorder to (C, H, W)

        if mask.shape[0] != 4:
            mask = mask.permute(2,0,1)
        mask = [:2,:,:]

        # Apply transformations
        if self.transform:
            # Keep track of the state so that the same transformation can be applied to both
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        # Convert mask to binary (if required)
        mask = torch.where(mask > 0, 1, 0).float()

        return image, mask, img_path  # Return the path for tracking purposes

# Example usage
if __name__ == "__main__":
    root_dir = "/content/"

    # Define any transformations you want to apply
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize if needed
    ])

    # Create the dataset and dataloader
    dataset = NumpyDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Iterate through the dataset
    for images, masks, paths in dataloader:
        print(f"Image paths: {paths}")
        print(f"Images batch shape: {images.size()}")
        print(f"Masks batch shape: {masks.size()}")