import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import pytorch_lightning as pl


#hyperparameters
batch_size = 32
epoch = 2
num_classes = 10
dataset_path = "/EuroSAT_RGB"

#parameters for data preprocessing
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

#load dataset
eurosat_dataset = ImageFolder(root=dataset_path, transform=transform)

#split dataset
train_size = int(0.8 * len(eurosat_dataset))
val_size = (len(eurosat_dataset) - train_size) // 2
test_size = len(eurosat_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    eurosat_dataset, [train_size, val_size, test_size]
)

#data loaders

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #add num_workers=... later
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=32)

class LandCoverModel(pl.LightningModule):
    def __init__(self, num_classes=num_classes):
        super(LandCoverModel, self).__init__()
        