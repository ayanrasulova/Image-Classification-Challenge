import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import cv2
from torchvision import transforms, models
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# (no gpu on mac, use device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class for data set
class MyDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.filenames = labels["filename"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        path = os.path.join(self.image_dir, fn)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label_a = self.labels.iloc[idx]["label_a"]
        label_b = self.labels.iloc[idx]["label_b"]
        label_idx = labels_to_numbers(label_a, label_b)
        return img, label_idx

# helper functions
def labels_to_numbers(label_a, label_b):
    if label_a == label_b:
        return label_a
    smaller = min(label_a, label_b)
    larger = max(label_a, label_b)
    return 10 + (larger * (larger - 1) // 2) + smaller

def number_to_labels(num):
    for i in range(10):
        for j in range(i, 10):
            if labels_to_numbers(i, j) == num:
                return (i, j)
    return None

# paths
train_image_path = "/content/data/train_set/blended_dataset/train/images"
labels = pd.read_csv("/content/data/train_set/blended_dataset/train/labels.csv")

# train transform 
train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# validation transform
val_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# train validation split
train_df, val_df = train_test_split(labels, test_size=0.1, shuffle=True, random_state=42)

train_dataset = MyDataset(train_image_path, train_df, transform=train_transform)
val_dataset   = MyDataset(train_image_path, val_df,   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

# model 
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 55)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop 

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_imgs, batch_labels in train_loader:
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_imgs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_imgs.size(0)

    avg_loss = running_loss / len(train_loader.dataset)

    # validation accuracy for each epoch
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_imgs, batch_labels in val_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_imgs)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")


# function for test dataset
class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.filenames = sorted(os.listdir(root))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        path = os.path.join(self.root, fn)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fn

# matching transform to validation
test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

test_path = "/content/data/test_set/blended_dataset/test/images"

test_dataset = TestDataset(test_path, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# MAKING THE CSV 

model.eval()
pred_rows = []

with torch.no_grad():
    for imgs, names in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        for fn, p in zip(names, preds):
            # convert predicted number to (label_a, label_b)
            label_a, label_b = number_to_labels(int(p))
            pred_rows.append((fn, label_a, label_b))

# saving the csv
df = pd.DataFrame(pred_rows, columns=["filename", "label_a", "label_b"])
save_path = "/content/drive/MyDrive/Machine-Learning-Contest/test_predictions.csv"
df.to_csv(save_path, index=False)

print("Saved predictions to:", save_path)
