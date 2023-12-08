import os
import random
import shutil
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import (
    ColorJitter,
    RandomGrayscale,
    RandomInvert,
    RandomPerspective,
    RandomRotation,
    RandomSolarize,
)

# hyperparameters
batch_size = 4
epoch = 10
num_classes = 10
dataset_path = "EuroSAT_RGB/"
model_checkpoint_path = "checkpoints/best_model.ckpt"

# parameters for data preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ]
)

# make dataset more diverse by rotating each training image
augmented_transform = transforms.Compose(
    [
        RandomRotation(degrees=180),
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ]
)

threshold = random.randint(0, 256)

solarize_transform = transforms.Compose(
    [
        RandomSolarize(threshold=threshold, p=1),
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ]
)


invert_transform = transforms.Compose(
    [
        RandomInvert(p=1),
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ]
)

dist_scale = random.random()
brightness = random.random()
contrast = random.random()
saturation = random.random()
hue = (0.0, 0.5)

color_jitter_transform = transforms.Compose(
    [
        ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        ),
        RandomPerspective(distortion_scale=dist_scale, p=1),
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ]
)

grayscale_transform = transforms.Compose(
    [
        RandomGrayscale(p=1),
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ]
)

# load dataset
eurosat_dataset = ImageFolder(root=dataset_path, transform=transform)

augmented_dataset = ImageFolder(root=dataset_path, transform=augmented_transform)

solarize_dataset = ImageFolder(root=dataset_path, transform=solarize_transform)

invert_dataset = ImageFolder(root=dataset_path, transform=invert_transform)

color_jitter_dataset = ImageFolder(root=dataset_path, transform=color_jitter_transform)

grayscale_dataset = ImageFolder(root=dataset_path, transform=grayscale_transform)


# split dataset
train_size = int(0.8 * len(eurosat_dataset))
val_size = (len(eurosat_dataset) - train_size) // 2
test_size = len(eurosat_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    eurosat_dataset, [train_size, val_size, test_size]
)

# concatenate datasets for more diversity
train_dataset = torch.utils.data.ConcatDataset(
    [
        eurosat_dataset,
        augmented_dataset,
        solarize_dataset,
        invert_dataset,
        color_jitter_dataset,
        grayscale_dataset,
    ]
)


# data loaders
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)  # add num_workers=... later
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=32)


# GELU performt eigentlich so ziemlich genau wie RELU, also man könnte auch ganz standard relu verwenden...
class LandCoverModel(pl.LightningModule):
    def __init__(self, num_classes=num_classes):
        super(LandCoverModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.GELU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.GELU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.GELU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.pool_output_size = 128 * (16 // 8) * (16 // 8)

        self.fc1 = nn.Linear(self.pool_output_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

        self.accuracy = Accuracy(num_classes=num_classes, task="multiclass")

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        # print("Shape before flattening:", x.shape)

        # dynamic calculation of the flattend tensors size
        # x_size = x.size(1) * x.size(2) * x.size(3)

        x = x.view(x.size(0), -1)

        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=1e-4, weight_decay=1e-5
        )  # original weight decay = 1e-5
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.01,
            total_steps=len(train_loader) * epoch,
            pct_start=0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = nn.CrossEntropyLoss()(outputs, y)
        acc = self.accuracy(outputs, y)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = nn.CrossEntropyLoss()(outputs, y)
        acc = self.accuracy(outputs, y)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)
        return loss


# create model
model = LandCoverModel(num_classes=len(eurosat_dataset.classes))


def train_model(model):
    valid_input = False
    while not valid_input:
        response = input("Do you want to train the model again? y/n")
        if response.lower() == "y":
            valid_input = True
        elif response.lower() == "n":
            valid_input = True
        else:
            print("Invalid input, please enter 'y' for yes, or 'n' for no")
    if response.lower() == "y":
        if os.path.isfile("checkpoints"):
            shutil.rmtree("checkpoints/")
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath="checkpoints",
            filename="best_model",
            save_last=True,
            save_top_k=1,
        )
        # training loop
        tb_logger = pl.loggers.TensorBoardLogger("logs/", name="land_cover_model")

        trainer = pl.Trainer(
            max_epochs=epoch, logger=tb_logger, callbacks=[checkpoint_callback]
        )
        trainer.fit(model, train_loader, val_loader)

        # save
        trainer.save_checkpoint(model_checkpoint_path)
    elif response.lower() == "n" and os.path.isfile("checkpoints/best_model.ckpt"):
        model = LandCoverModel.load_from_checkpoint(
            checkpoint_path=model_checkpoint_path
        )
    else:
        print("no pretrained model found")


# test own image and classify it
def test_own_image():
    response = input("Enter custom image name:")
    if os.path.isfile("images/" + response):
        image_path = "images/" + response
        input_image = Image.open(image_path).convert("RGB")
        input_tensor = transform(input_image)
        input_batch = input_tensor.unsqueeze(0)

        model.eval()

        with torch.no_grad():
            output = model(input_batch)

        # get predicted class label
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.item()
        predicted_class_dict = {
            0: "Annual Crop",
            1: "Forest",
            2: "Herbaceous Vegetation",
            3: "Highway",
            4: "Industrial",
            5: "Pasture",
            6: "Permanent Crop",
            7: "Residential",
            8: "River",
            9: "Sea/Lake",
        }
        print(
            f"The predicted class for your input Image is: {predicted_class_dict[predicted_class]}"
        )
        test_own_image()
    else:
        print("image not found, try again!")
        test_own_image()


train_model(model)
test_own_image()
