import shutil
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchmetrics.classification import Accuracy
import pytorch_lightning as pl
from PIL import Image
import os


#hyperparameters
batch_size = 2
epoch = 1
num_classes = 10
dataset_path = "EuroSAT_RGB/"
model_checkpoint_path = "checkpoints/best_model.ckpt"

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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

        self.accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1 ,128 * 32 * 32)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = nn.CrossEntropyLoss()(outputs, y)
        acc = self.accuracy(outputs, y)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = nn.CrossEntropyLoss()(outputs, y)
        acc = self.accuracy(outputs, y)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)
    
#create model
model = LandCoverModel(num_classes=len(eurosat_dataset.classes))

def train_model():
    valid_input = False
    while not valid_input:
        response = (input("Do you want to train the model again? y/n"))
        if response.lower() == 'y':
            valid_input = True
        elif response.lower() == 'n':
            valid_input = True
        else:
            print("Invalid input, please enter 'y' for yes, or 'n' for no")
    if response.lower() == 'y':
        if os.path.isfile("checkpoints"):
            shutil.rmtree("checkpoints/")
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath="checkpoints",
            filename="best_model",
            save_last=True,
            save_top_k=1
        )
        #training loop
        tb_logger = pl.loggers.TensorBoardLogger("logs/", name="land_cover_model")

        trainer = pl.Trainer(max_epochs=epoch, logger=tb_logger, callbacks=[checkpoint_callback])
        trainer.fit(model, train_loader, val_loader)

        #save
        trainer.save_checkpoint(model_checkpoint_path)
    elif response.lower() == 'n' and os.path.isfile("checkpoints/best_model.ckpt"):
        model = LandCoverModel.load_from_checkpoint(checkpoint_path=model_checkpoint_path)
    else:
        print("no pretrained model found")

#test own image and classify it
def test_own_image():
    response = (input("Enter custom image name:"))
    if os.path.isfile("images/" + response):
        image_path = "images/" + response
        input_image = Image.open(image_path).convert("RGB")
        input_tensor = transform(input_image)
        input_batch = input_tensor.unsqueeze(0)

        model.eval()

        with torch.no_grad():
            output = model(input_batch)
        
        #get predicted class label
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.item()
        print(f"The predicted class for your input Image is: {predicted_class}") #add dictionary instead of numbers
        test_own_image()
    else:
        print("image not found, try again!")
        test_own_image()

train_model()
test_own_image()