from spawn_classifier import SC_Dataset, SC_Model
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
import sys


# Initialize our Transpose
t = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# initialize our dataset and splitters
data = np.loadtxt("data/main.txt")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# train the data on a certain index. Then evaluate with the validation idx
for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
    train = data[train_idx]
    test = data[val_idx]

    # initialize both loaders
    train = SC_Dataset.SC_Dataset(train, transform=None)
    test = SC_Dataset.SC_Dataset(test, transform=None)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    val_loader   = DataLoader(test, batch_size=64)

    # initialize simple model
    model = SC_Model.SimpleCNN()
    num_epochs = 2
    
    # train the model 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    
    # evaluate the model 
    # model.eval()
    # running_loss = 0.0
    # with torch.no_grad:
    #     for images, labels in val_loader:
    #         outputs = model(data)
    #         loss = criterion(outputs, labels)
    #         running_loss += loss.item() * inputs.size(0)
    #     val_loss = running_loss / len(val_loader.dataset)
    #     val_losses.append(val_loss)
        
sys.exit(1)

