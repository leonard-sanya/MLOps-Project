import os
import torch
import timm
import torchmetrics
from images_dataset import ImageDataset, class_mapping
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

batch_size = 32
num_workers = 4

train_data, val_data, test_data = torch.utils.data.random_split(images, [train_num, val_num, test_num])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

classes = sorted(class_mapping.keys())

model = timm.create_model("rexnet_150", pretrained=True, num_classes=len(classes))


def to_device(batch, device):
    return batch[0].to(device), batch[1].to(device)


def get_metrics(model, ims, gts, loss_fn, epoch_loss, epoch_acc, epoch_f1, f1_score):
    preds = model(ims)
    loss = loss_fn(preds, gts)
    epoch_loss += loss.item()
    epoch_acc += (torch.argmax(preds, dim=1) == gts).sum().item()
    epoch_f1 += f1_score(preds, gts).item()
    return loss, epoch_loss, epoch_acc, epoch_f1

