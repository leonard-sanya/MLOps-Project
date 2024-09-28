import os
import torch
import timm
import torchmetrics
from images_dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

root = 'hollywood_data'

mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],224
transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

images = ImageDataset(root_dir=root, transform=transform)
num_images = len(images)
train_num = int(num_images * 0.9)
val_num = int(num_images * 0.05)
test_num = num_images - train_num - val_num

train_data, val_data, test_data = torch.utils.data.random_split(images, [train_num, val_num, test_num])

