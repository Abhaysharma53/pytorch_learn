import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv('fmnist_small.csv')

# fig, axes = plt.subplots(4, 4, figsize = (10, 10))

# divide the data into feature and target
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

#scaling the feature 
X_train = X_train/255.0
X_test = X_test/255.0

# create CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values, dtype = torch.float32)
        self.labels = torch.tensor(labels.values, dtype = torch.long)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

#dataloader object
train_loader = DataLoader(train_dataset, batch_size= 64, shuffle= True, pin_memory= True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle= False, pin_memory=True)

class MyNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.Model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)

        )
    def forward(self, X):
       return self.Model(X)
    
#set learning rate and epochs
learning_rate = 0.1
epochs = 100

# instantiate model, loss function and optimizer
model = MyNN(X_train.shape[1])
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

#training loop
for epoch in range(epochs):
    total_epoch_loss = 0 
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        #forward pass
        pred = model(batch_features)
        #calculate loss
        loss = criterion(pred, batch_labels)
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        #update weights
        optimizer.step()
        total_epoch_loss = total_epoch_loss + loss.item()
    avg_loss = total_epoch_loss/len(train_loader)
    print(f'Epoch - {epoch +1 }, Loss - {avg_loss}')
    
#set model to eval mode
model.eval()

# model evaluation
with torch.no_grad():
    total = 0
    correct = 0
    for batch_features, batch_labels in test_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        pred = model(batch_features)
        _, predicted = torch.max(pred, 1)
        total = total + batch_labels.shape[0]
        correct = correct + (predicted == batch_labels).sum().item()
    print(correct / total)









    





