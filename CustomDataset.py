import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F, Compose, ToTensor

class FastRotationLargeAngles:
    def __init__(self, angle_range=(-90, 90), num_angles=100):
        """
        Fast rotation with a precomputed large number of angles.
        Args:
            angle_range (tuple): The range of angles (min_angle, max_angle).
            num_angles (int): The number of discrete angles to precompute.
        """
        self.angles = torch.linspace(angle_range[0], angle_range[1], steps=num_angles).tolist()

    def __call__(self, image, idx=None):
        """
        Apply rotation from precomputed angles.
        Args:
            image (PIL Image): The input image.
            idx (int): Index for deterministic behavior (optional).
        """
        if idx is None:
            idx = torch.randint(0, len(self.angles), (1,)).item()  # Random selection
        angle = self.angles[idx % len(self.angles)]  # Cycle through angles
        return F.rotate(image, angle)

class NonAugmentedDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (str): Path to the dataset folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.temperatures = []

        # Load image paths and temperatures
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".tiff"):
                self.image_paths.append(os.path.join(folder_path, filename))
                temperature = float(filename.split('_')[-1].replace('C.tiff', ''))
                self.temperatures.append(temperature)

        # Convert temperatures to a torch tensor
        self.temperatures = torch.tensor(self.temperatures, dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image_array = np.array(image)
        image_tensor = torch.tensor(image_array, dtype=torch.float32)

        # Add a channel dimension if image is grayscale
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)

        # Normalize image
        image_tensor = image_tensor / 255.0  # Scale to [0, 1]

        # Apply any specified transformations
        if self.transform:
            image_tensor_2 = self.transform(image_tensor)
        else:
            image_tensor_2 = image_tensor

        # Get the corresponding temperature
        temperature = self.temperatures[idx]

        return (image_tensor, image_tensor_2), temperature

class AugmentedDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (str): Path to the dataset folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.temperatures = []

        # Parse dataset folder for images and temperatures
        for subdir, _, files in os.walk(folder_path):
            for filename in sorted(files):
                if filename.endswith(".tiff"):
                    temperature = float(filename.split('_')[3].replace('¯C', ''))
                        
                    # Append the image and its temperature
                    self.image_paths.append(os.path.join(subdir, filename))
                    self.temperatures.append(temperature)

        # Convert temperatures to a torch tensor
        self.temperatures = torch.tensor(self.temperatures, dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image_array = np.array(image, dtype=np.float32)
        image_tensor = torch.tensor(image_array, dtype=torch.float32)

        # Add a channel dimension if image is grayscale
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)

        # Normalize image to [0, 1]
        image_tensor = image_tensor / 255.0

        # Apply any specified transformations
        if self.transform:
            image_tensor_2 = self.transform(image_tensor)
        else:
            image_tensor_2 = image_tensor

        # Get the corresponding temperature
        temperature = self.temperatures[idx]

        return (image_tensor, image_tensor_2), temperature