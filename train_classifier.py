from spawn_classifier import SC_Dataset, SC_Model
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image


print("The Start")
t = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = SC_Dataset.SC_Dataset(f'main', transform=None)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print("Initialized Data Loaders")

model = SC_Model.SimpleCNN()
num_epochs = 2
print("Initialized Model")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()
print("Initialized Training")

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
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
        if i % 100 == 99:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0