import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import clip
import torch.nn as nn
import torch.optim as optim

# Step 1: Define the custom dataset
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.class_to_idx = {
            'Normal': 0,
            'Mild': 1,
            'Moderate': 2,
            'Severe': 3,
            'Proliferate': 4
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 1]
        image = Image.open(img_name).convert("RGB")
        label_str = self.data.iloc[idx, 2]
        label = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.train()
# Create Dataset and DataLoader
dataset = DiabeticRetinopathyDataset(csv_file='/teamspace/studios/this_studio/diabetic_retinopathy/funds_train_data.csv',
                                     transform=preprocess)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Replace CLIP's head with a custom classifier for 5 classes
model.visual.fc = nn.Sequential(
    nn.Linear(model.visual.output_dim, 5)  # 5 classes: No_DR, Mild, Moderate, Severe, Proliferative_DR
).to(device)

# Step 4: Define Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.visual.fc.parameters(), lr=0.1)

from tqdm import tqdm
# model = model.float()

def train_model(model, dataloader, criterion, optimizer, num_epochs=5):
    model.train()

    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
            f.write(f'Epoch {epoch+1}/{num_epochs}\n')

        # Wrap the dataloader with tqdm for batch progress
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model.visual(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()

                # Update the progress bar with loss and accuracy for the current batch
                tepoch.set_postfix(loss=running_loss / (tepoch.n + 1), accuracy=100 * correct / total)

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
            f.write(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}% \n\n')

        torch.save(model.state_dict(), '/teamspace/studios/this_studio/diabetic_retinopathy/clip_diabetic_retinopathy_model.pth')
    print('Training complete')

# Step 6: Train the model
train_model(model, dataloader, criterion, optimizer, num_epochs=30)

# Step 7: Save the trained model
