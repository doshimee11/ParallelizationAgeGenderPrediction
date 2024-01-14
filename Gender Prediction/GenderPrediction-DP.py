import pandas as pd
import numpy as np
import tarfile
import io
import os

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split
from skimage import io
import PIL
import torch

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Compose
import torch.nn.functional as F
from IPython.display import display

# PyTorch libraries and modules
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.optim as optim

from timeit import default_timer as timer


class XRayTrain(Dataset):
    def __init__(self, csv_file, img_dir, transform = None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(csv_file, nrows = 10702, dtype = {'imageId': str})
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, f"{self.annotations.iloc[index, 0]}.png")
        img = Image.open(img_path)
        img = img.resize((resize_value, resize_value))
        img = Image.fromarray(255 - np.array(img))
        y_label = torch.tensor(int (self.annotations.iloc[index, 1]))

        if self.transform:
            img = self.transform(img)

        return (img, y_label)


# Creating the Gender Model
class GenderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = (3, 3), stride = 1, padding = 1)
        self.batch1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = (3, 3), stride = 1, padding = 1)
        self.batch2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = (3, 3), stride = 1, padding = 1)
        self.batch3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size = (3, 3), stride = 1, padding = 1)
        self.batch4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.drop1 = nn.Dropout(p = 0.4)
        self.fc2 = nn.Linear(1024, 128)
        self.drop2 = nn.Dropout(p = 0.5)
        self.fc3 = nn.Linear(128, 2)

    def forward(self,x):
        x = self.pool1(F.relu(self.batch1((self.conv1(x)))))
        x = self.pool2(F.relu(self.batch2((self.conv2(x)))))
        x = self.pool3(F.relu(self.batch3((self.conv3(x)))))
        x = self.pool4(F.relu(self.batch4((self.conv4(x)))))
        x = x.view(-1, 128 * 8 * 8)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


test_folder = '/home/doshi.mee/csye7105_project/dataset/SPR X-Ray Images/kaggle/kaggle/test/'
train_folder = '/home/doshi.mee/csye7105_project/dataset/SPR X-Ray Images/kaggle/kaggle/train/'

train_age_csv = '/home/doshi.mee/csye7105_project/dataset/SPR X-Ray Images/train_age.csv'
train_gender_csv = '/home/doshi.mee/csye7105_project/dataset/SPR X-Ray Images/train_gender.csv'

resize_value = 128
batch_size = 128
epochs = 6


# Creating tensors for the images and their labels for gender
train_g_data = XRayTrain(csv_file = train_gender_csv, img_dir = train_folder,
                       transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                        transforms.Resize((resize_value, resize_value)),
                                                        transforms.ToTensor()]))

train_size = int(0.8 * len(train_g_data))
test_size = len(train_g_data) - train_size

train_gender_data, test_gender_data = torch.utils.data.random_split(train_g_data, [train_size, test_size])


# loading the data
train_gender_loader = DataLoader(dataset = train_gender_data, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
test_gender_loader = DataLoader(dataset = test_gender_data, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)


CUDA_LAUNCH_BLOCKING =1
torch.cuda.empty_cache()

# Set the number of GPUs required
gpus = 4

model1 = GenderModel()
if gpus >= 1:
    print("Let's use", gpus, "GPUs!")
    model = nn.DataParallel(model1, list(range(gpus)))

device = torch.device("cuda")
model.to(device)


def train(): 
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)   
    
    n_samples = 10240
    total_loss = 0
    total_train = 0
    correct_train = 0.0
    
    for i, (img, labels) in enumerate(train_gender_loader):
        img = img.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        
        #accuracy
        _, predicted = torch.max(output.data, 1)
        predicted = predicted
        total_train += labels.size(0)
        correct_train += predicted.eq(labels.data).sum().item()
        train_accuracy = 100 * correct_train / total_train
        
    losses.append(loss.item())
    train_acy.append(train_accuracy)
    return (train_accuracy, loss.item())


def valid(epochs):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (img, labels) in enumerate(test_gender_loader):
                img = img.to(device)
                labels = labels.to(device)
                outputs = model(img)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100 * correct / total
                
    valid_acy.append(acc)
    return acc


print("Training for {} epochs".format(epochs))

main_time = timer()
losses = []
train_acy = []
valid_acy = []

for epoch in range(epochs):
    start = timer()
    (train_acc, loss_tr) = train()
    valid_acc = valid(epochs)
    
    print('Epoch {}, train Loss: {:.3f}'.format(epoch ,loss_tr), 'Training Accuracy {}:' .format(train_acc), 'Valid Accuracy {}'.format(valid_acc))
    print("GPU time: ", timer() - start)

folder = '/home/doshi.mee/csye7105_project'
torch.save(model, f"{folder}/gender_model_dp.pt")
print("Model successfully saved")
torch.cuda.empty_cache()

print("Time taken for", epochs, "epochs is", timer() - main_time, "seconds")
