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

def ReadEncoderData(path : str)->None:

  def numpy_read(path : str):
    loss_values = np.load(path)
    return np.arange(1, len(loss_values)+1), loss_values

  def string_read(path : str):
    loss_values = []

    with open(path, 'r') as file:
      for line in file:
        if "Loss:" in line and "Validation" not in line:
          loss_values.append(float(line.split("Loss:")[1].strip()))
    loss_values = np.array(loss_values)
    num_epochs_r = np.arange(1, len(loss_values)+1)

    return num_epochs_r, loss_values
  
  for filename in sorted(os.listdir(path)):
    if filename.endswith(".txt"):
      return string_read(os.path.join(path, filename))
    elif filename.endswith(".npy"):
      return numpy_read(os.path.join(path, filename))
  raise ValueError("Data does not exist in path")

def ReadRegressorData(path : str):
  def string_read(path : str):
    mse_values = []
    validation_mse_values = []
    validation_r2_values = []

    with open(path, 'r') as file:
      for line in file:
        if "MSE :" in line and "Validation" not in line:
          mse_values.append(float(line.split("MSE :")[1].strip()))
        elif "Validation: MSE :" in line:
          validation_mse_values.append(float(line.split("Validation: MSE :")[1].strip()))
        elif "Validation R2:" in line:
          validation_r2_values.append(float(line.split("Validation R2:")[1].strip()))
    mse_values = np.array(mse_values)
    validation_mse_values = np.array(validation_mse_values)
    validation_r2_values = np.array(validation_r2_values)
    num_epochs_r = np.arange(1, len(mse_values)+1)

    return num_epochs_r, mse_values, validation_mse_values, validation_r2_values
  
  for filename in sorted(os.listdir(path)):
    if filename.endswith(".txt"):
      return string_read(os.path.join(path, filename))
    elif filename.endswith("train_losses.npy"):
      mse_values = np.load(os.path.join(path, filename))
    elif filename.endswith("val_losses.npy"):
      validation_mse_values = np.load(os.path.join(path, filename))
    elif filename.endswith("val_r2.npy"):
      r2_values = np.load(os.path.join(path, filename))
  
  return np.arange(1, len(mse_values)+1), mse_values, validation_mse_values, r2_values

def DataLoader2Numpy(dataloader : DataLoader):
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

def GetExtremelyNoisyDataset(data_augmentation : bool, device : str = "cuda:0"):
    # data_augmentation isn't used, just to keep it consistent
    # Load Kaggle dataset
    path = kagglehub.dataset_download("juanda220485/synthetic-dataset-of-speckle-images")
    print("Path to dataset files:", path)

    transforms = torchvision.transforms.Compose([
      lambda x : x.unsqueeze(0),
      GaussianNoise(mean=128.0, sigma=(1, 30), clamp=255, device=device),
      lambda x : x / 255.0,
      ImageMask(sz=(126, 126), device=device),
      RandomRotationTensor(degrees=360, samples=128),
    ])
      
    return CustomDataset(path, transform=torchvision.transforms.Compose([
        MultiView(transforms, n_views=1)
    ]), device=device
    )

def shufflenetv2_x1_model_runner_get_basic_model():
  from torchvision.models import shufflenet_v2_x1_0

  class SupConModel(nn.Module):
    def __init__(self, feature_dim : int = 128, device : str = "cpu")->None:
      super(SupConModel, self).__init__()
      self.backbone = shufflenet_v2_x1_0(pretrained=True)
      self.backbone.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1)
      self.backbone.classifier     = nn.Identity().to(device)
      self.linear = nn.Linear(1000, feature_dim)
      self.encoder = nn.Sequential(
        self.backbone,
        self.linear
      ).to(device)

    def forward(self, X):
      X = self.encoder(X)
      X = F.normalize(X, dim=1)

      return X

  class RegressionModel(torch.nn.Module):
    def __init__(self, supconmodel : SupConModel, features_dim : int = 128, device : str = 'cpu')->None:
      super(RegressionModel, self).__init__()
      self.encoder = supconmodel.encoder

      for param in self.encoder.parameters():
        param.requires_grad = False

      self.head = nn.Sequential (
        nn.Linear(features_dim, features_dim//2),
        nn.ReLU(),
        nn.Linear(features_dim//2, 1),
      ).to(device)

    def forward(self, X):
      with torch.no_grad():
        X = self.encoder(X)
      X = self.head(X)

      return X
    
  return SupConModel, RegressionModel

def resnet18_baseline_model_runner_get_basic_model():
  from torchvision.models import resnet18

  class ResNet18E(nn.Module):
    def __init__(self, device : str = "cpu")->None:
      super(ResNet18E, self).__init__()
      self.backbone = resnet18(pretrained=True)
      self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.backbone.fc    = nn.Identity()
      self.linear = nn.Linear(512, 1)
      self.encoder = nn.Sequential(
        self.backbone,
        self.linear
      ).to(device)

    def forward(self, X):
      return self.encoder(X)
    
  return ResNet18E

def resnet18_model_runner_get_basic_models():
  from torchvision.models import resnet18

  class SupConModel(nn.Module):
    def __init__(self, feature_dim : int = 128, device : str = "cpu")->None:
      super(SupConModel, self).__init__()
      self.backbone = resnet18(pretrained=True)
      self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
      self.backbone.fc    = nn.Identity()
      self.linear = nn.Linear(512, feature_dim)
      self.encoder = nn.Sequential(
        self.backbone,
        self.linear
      ).to(device)
      self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
      X = self.encoder(X)
      X = self.softmax(X)

      return X
  
  class RegressionModel(torch.nn.Module):
    def __init__(self, supconmodel : SupConModel, features_dim : int = 128, device : str = 'cpu')->None:
      super(RegressionModel, self).__init__()
      self.encoder = supconmodel.encoder

      for param in self.encoder.parameters():
        param.requires_grad = False

      self.head = nn.Sequential (
        nn.Linear(features_dim, features_dim//2),
        nn.ReLU(),
        nn.Linear(features_dim//2, 1),
      ).to(device)

    def forward(self, X):
      with torch.no_grad():
        X = self.encoder(X)
      X = self.head(X)
      return X
  
  return SupConModel, RegressionModel

def model_runner_interface(path : str, noisy_dataset : bool = False, N : int = 1):
  from torch.utils.data import random_split
  
  model = torch.load(path)
  split_ratio = 0.9
  batch_size = 64
  sz = (126, 126)

  # get dataset
  dataset = GetExtremelyNoisyDataset(data_augmentation=False) if noisy_dataset else \
            GetDataset(data_augmentation=False)

  # Get the total size of the dataset
  dataset_size = len(dataset)

  # Calculate sizes for training and validation splits
  train_size = int(split_ratio * dataset_size)
  val_size = dataset_size - train_size

  # Perform the split
  generator1 = torch.Generator().manual_seed(42)
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator1)

  # Create DataLoaders for training and validation
  #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True )
  val_loader   = DataLoader(val_dataset  , batch_size=batch_size, shuffle=False)

  model.eval()
  with torch.no_grad():
    y_pred = []
    y_test = []
    for _ in range(N):
      for images, temperatures in val_loader:
        N = images.shape[0]
        V = images.shape[1]
        images = images.reshape(N * V, 1, sz[0], sz[1])
        temperatures = temperatures.unsqueeze(1).expand(-1, V).reshape(N * V, 1)
        pred = model(images)

        y_pred.append(pred.squeeze().cpu().detach().numpy())
        y_test.append(temperatures.squeeze().cpu().detach().numpy())
      
  y_pred = np.array(y_pred).flatten()
  y_test = np.array(y_test).flatten()

  return y_pred, y_test

def sklearn_regression_model_runner(model, noisy_dataset : bool = False, N : int = 1):
  from torch.utils.data import random_split

  split_ratio = 0.9
  batch_size  = 1
  dataset = GetExtremelyNoisyDataset(data_augmentation=False) if noisy_dataset else \
            GetDataset(data_augmentation=False)

  # Get the total size of the dataset
  dataset_size = len(dataset)

  # Calculate sizes for training and validation splits
  train_size = int(split_ratio * dataset_size)
  val_size = dataset_size - train_size

  # Perform the split
  generator1 = torch.Generator().manual_seed(42)
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator1)

  # Create DataLoaders for training and validation
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True )
  val_loader   = DataLoader(val_dataset  , batch_size=batch_size, shuffle=False)

  Xt, yt   = DataLoader2Numpy(train_loader)
  Xt = Xt.reshape(len(Xt), np.prod(Xt.shape[1:]))

  model.fit(Xt, yt)

  y_pred = []
  y_test = []
  for _ in range(N):
    Xv, yv   = DataLoader2Numpy(val_loader)
    Xv = Xv.reshape(len(Xv), np.prod(Xv.shape[1:]))

    pred = model.predict(Xv)
    y_pred.append(pred)
    y_test.append(yv)

  y_pred = np.array(y_pred).flatten()
  y_test = np.array(y_test).flatten()

  return y_pred, y_test

class PairDet:
  def __init__(self, threshold : float = 1.0):
    self.T = threshold

  def __call__(self, y : torch.Tensor)->torch.Tensor:
    # A positive pair is when |yi - yj| <= T, were i, j in N
    # This returns a square matrix of (N, N) where
    # each row is an image, and each column where Aij = 1 is a positive pair
    #

    # replicate y, such that [y, y, ..., y] for N columns
    N  = len(y)
    Y  = y.expand(-1, N)
    Q  = torch.abs(Y - Y.T) <= self.T

    return Q