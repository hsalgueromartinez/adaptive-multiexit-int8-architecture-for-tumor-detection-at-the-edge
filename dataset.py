import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets import PCAM

class RandomDiscreteRotation:
    """Randomly rotate image by 0, 90, 180, or 270 degrees for microscopy invariance."""
    def __init__(self, angles=(0, 90, 180, 270)):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return TF.rotate(img, angle)

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# Pipeline A: Minimalist baseline (no domain-specific augmentations)
raw_train_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# Pipeline B: Histology-informed (geometric invariance + stain simulation)
augmented_train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    RandomDiscreteRotation([0, 90, 180, 270]),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.04),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# Evaluation / Validation / Test: Strictly deterministic
eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

def get_pcam_loaders(data_root="./data", pipeline_type="augmented", batch_size=64, num_workers=2, subset_fraction=1.0):
    """
    Loads PCam splits.
    pipeline_type: 'raw' or 'augmented'
    subset_fraction: fraction of dataset to use for fast local validation (e.g. 0.1 for 10% data)
    """
    train_tf = augmented_train_transform if pipeline_type == "augmented" else raw_train_transform
    
    train_set = PCAM(root=data_root, split="train", transform=train_tf, download=True)
    val_set   = PCAM(root=data_root, split="val",   transform=eval_transform,  download=True)
    test_set  = PCAM(root=data_root, split="test",  transform=eval_transform,  download=True)

    if subset_fraction < 1.0:
        n_train = int(len(train_set) * subset_fraction)
        n_val = int(len(val_set) * subset_fraction)
        train_set = torch.utils.data.Subset(train_set, range(n_train))
        val_set = torch.utils.data.Subset(val_set, range(n_val))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader