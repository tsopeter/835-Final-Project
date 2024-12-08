import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as vF
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import kagglehub
import numpy as np
import os
from PIL import Image
import random

def DataLoader2Numpy(dataloader : DataLoader)->np.ndarray:
  X = []
  y = []
  for images, temps in dataloader:
    N = images.shape[0]
    V = images.shape[1]
    S = images.shape[-1]
    images = images.reshape(N * V, S, S)
    temps  = temps.reshape(N * V)
    
    X.append(images.cpu().detach().numpy())
    y.append(temps.cpu().detach().numpy())

  X, y = np.array(X), np.array(y)
  X = X.reshape(np.prod(X.shape[0:2]), S, S)
  y = y.flatten()
  return X, y

class ImageMask():
  def __init__(self, sz : tuple, fit : bool = True, device : str = "cuda:0")->None:
    """
    Args:
        sz (tuple): Size of image
        fit (bool, optional): Fit diameter to smaller dimensions of sz
    """
    self.height = sz[0]
    self.width  = sz[1]
    self.radius = (self.height if self.height < self.width else self.width) / 2
    self.device = device

    y, x = np.ogrid[:self.height, :self.width]

    # Calculate the center of the circle
    center_y, center_x = self.height // 2, self.width // 2

    # Create the mask for the circle
    distance_from_center = (y - center_y)**2 + (x - center_x)**2
    self.mask = distance_from_center <= self.radius**2

    self.mask = torch.tensor(self.mask, device=device)

  def __call__(self, image):
    mask = self.mask
    if len(image.shape) == 3:
      mask = mask.unsqueeze(0)

    return image * mask

class RandomCrop():
  def __init__(self, sz : tuple, depth : int = 1)->None:
    """
    Args:
        sz (tuple): Size of cropped image
        depth (int, optional): Returns # of randomly cropped images
    """
    assert depth > 0
    self.depth = depth
    self.sz    = sz

    self.cropper = torchvision.transforms.RandomCrop(self.sz)

  def __call__(self, image):
    if self.depth == 1:
      return self.cropper(image)

    # For depth > 1, apply cropping multiple times
    crops = [self.cropper(image) for _ in range(self.depth)]
    # Stack the crops along a new dimension to get shape (D, S, S)
    return torch.stack(crops)

class GaussianNoise():
  def __init__(self, mean : float | list | tuple = 0.0, sigma : float | list | tuple = 1.0, clamp : float = 1.0, mode : int = 0, device : str = "cuda:0")->None:
    """
    Args:
        mean (float, list, optional): mean of gaussian distribution
        sigma (float, list, optional): standard deviation of gaussian distribution
        mode (int, optional): Random (slower) or Iterative (faster)
    """
    self.device = device
    if isinstance(mean, tuple):
      self.means = np.linspace(mean[0], mean[1], 16)
    else:
      self.means  = np.array(mean)

    if isinstance(sigma, tuple):
      self.sigmas = np.linspace(sigma[0], sigma[1], 16)
    else:
      self.sigmas = np.array(sigma)

    # Pair up as [N, mean , sigma]
    self.pairs = np.array(np.meshgrid(self.means, self.sigmas)).T.reshape(-1, 2)
    self.samples = len(self.pairs)
    self.mode = mode
    self.idx = 0
    self.clamp = clamp

  def __call__(self, image):
    if self.mode == 0:
      idx = torch.randint(0, self.samples, (1,)).item()  # Random selection
    else:
      idx = self.idx
      self.idx += 1 # iterate idx

    params = self.pairs[idx % self.samples,:]  # Cycle through angles

    # Add Gaussian noise to the input tensor
    noisy_tensor = image + torch.normal(mean=params[0], std=params[1], size=image.size(), device=self.device)

    noisy_tensor = torch.clamp(noisy_tensor, 0.0, self.clamp)
    return noisy_tensor

class RandomFlip():
  def __init__(self, prob: float = 0.5):
    self.prob = prob

  def __call__(self, image):
    decision = torch.rand(1).item()
    if decision < (self.prob / 2):
      image = vF.vflip(image)
    elif decision < (self.prob):
      image = vF.hflip(image)
    
    return image

class RandomRotationTensor():
  def __init__(self, degrees : float | tuple | int, samples : int = 64, mode : int = 0)->None:
    """
    Args:
        degress (float, tuple): If float or int, then we use sample from -x to x. If it is a tuple, it is (min, max)
        samples (int, optional): Number of samples to select from (the precision is (stop - start)/samples)
    """
    if isinstance(degrees, float) or isinstance(degrees, int):
      start = -degrees
      stop  =  degrees

    if isinstance(degrees, tuple):
      start = degrees[0]
      stop  = degrees[1]

    self.degrees = np.linspace(start=start, stop=stop, num=samples)

    self.samples = samples
    self.mode    = mode
    self.idx     = 0

  def __call__(self, image):
      if self.mode == 0:
        idx = torch.randint(0, self.samples, (1,)).item()  # Random selection
      else:
        idx = self.idx
        self.idx += 1 # iterate idx

      angle = self.degrees[idx % self.samples]  # Cycle through angles
      return torchvision.transforms.functional.rotate(image, angle)

class MultiView():
  def __init__(self, transforms : any, n_views : int = 10):
    self.n_views = n_views
    self.transforms = transforms

  def __call__(self, image):
    if self.n_views == 1:
      return self.transforms(image)

    # For depth > 1, apply cropping multiple times
    views = [self.transforms(image) for _ in range(self.n_views)]
    # Stack the crops along a new dimension to get shape (D, S, S)
    return torch.stack(views)
  
class RandomCut():
  def __init__(self, sz : tuple, prob : float = 0.5):
    # randomly selects a portion of the image to cut
    self.sz = sz
    self.prob = prob

  def __call__(self, image):
    x = random.randint(0, 126-self.sz[0])
    y = random.randint(0, 126-self.sz[1])

    h = random.randint(10, self.sz[0])
    w = random.randint(10, self.sz[1])

    decision = torch.rand(1).item()
    if decision < self.prob:
      image[0, x:x+h, y:y+w] = 0
    return image

class CustomDataset(Dataset):
    def __init__(self, folder_path, device : str = "cpu", transform=None):
        """
        Args:
            folder_path (str): Path to the dataset folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_path  = folder_path
        self.transform    = transform
        self.image_paths  = []
        self.images       = []
        self.temperatures = []
        self.raw          = False
        self.device       = device

        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".tiff"):
                image_path = os.path.join(folder_path, filename)
                self.image_paths.append(image_path)
                self.images.append(np.array(Image.open(image_path)))
                temperature = float(filename.split('_')[-1].replace('C.tiff', ''))
                self.temperatures.append(temperature)

        self.temperatures = torch.tensor(self.temperatures, dtype=torch.float32, device=device)
        self.images       = torch.tensor(np.array(self.images), device=device)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image       = self.images[idx]
        temperature = self.temperatures[idx]

        if not self.raw:
          if self.transform is not None:
            image = self.transform(image)

        return image, temperature

    def numpy(self, raw : bool = True):
      # get as numpy array (after transforms)
      X = []
      y = []
      prev = self.raw
      self.raw = raw
      for i in range(len(self.images)):
        image, temperature = self.__getitem__(i)

        X.append(image.cpu().detach().numpy())
        y.append(temperature.cpu().detach().numpy())

      self.raw = prev
      return np.array(X), np.array(y)
    
def GetDataset(data_augmentation : bool = True, n_views : int = 4, device : str = "cuda:0"):
    # Load Kaggle dataset
    path = kagglehub.dataset_download("juanda220485/synthetic-dataset-of-speckle-images")
    print("Path to dataset files:", path)


    if data_augmentation:
        transforms = torchvision.transforms.Compose([
            lambda x : x.unsqueeze(0),
            GaussianNoise(mean=0.0, sigma=(1, 15), clamp=255, device=device),
            lambda x : x / 255.0,
            ImageMask(sz=(126, 126), device=device),
            #RandomCut((64, 64), prob=0.3),
            RandomFlip(prob=0.667),
            RandomRotationTensor(degrees=360, samples=128),
        ])
    else:
      transforms = torchvision.transforms.Compose([
            lambda x : x.unsqueeze(0),
            GaussianNoise(mean=0.0, sigma=(1, 15), clamp=255, device=device),
            lambda x : x / 255.0,
            ImageMask(sz=(126, 126), device=device),
            RandomRotationTensor(degrees=360, samples=128),
        ])
      


    return CustomDataset(path, transform=torchvision.transforms.Compose([
        MultiView(transforms, n_views=(n_views if data_augmentation else 1))
    ]), device=device
    )