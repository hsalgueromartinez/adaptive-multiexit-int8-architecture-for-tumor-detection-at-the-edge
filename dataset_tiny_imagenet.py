import os
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomCrop(64, padding=4),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

def get_tiny_imagenet_loaders(data_root="./data_tiny_imagenet/tiny-imagenet-200", batch_size=64, num_workers=2):
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")

    train_set = ImageFolder(train_dir, transform=train_transform)
    val_set   = ImageFolder(val_dir, transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader