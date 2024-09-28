import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class_mapping = {
    0: 'Angelina Jolie', 1: 'Brad Pitt', 2: 'Denzel Washington',
    3: 'Hugh Jackman', 4: 'Jennifer Lawrence', 5: 'Johnny Depp',
    6: 'Kate Winslet', 7: 'Leonardo DiCaprio', 8: 'Megan Fox',
    9: 'Natalie Portman', 10: 'Nicole Kidman', 11: 'Robert Downey Jr',
    12: 'Sandra Bullock', 13: 'Scarlett Johansson', 14: 'Tom Cruise',
    15: 'Tom Hanks', 16: 'Will Smith'
}


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(glob(os.path.join(root_dir, '*/*')))
        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = class_mapping

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[class_name]
        if self.transform:
            image = self.transform(image)

        return image, label
