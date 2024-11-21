import torch

from transformers import ViTFeatureExtractor

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd
train_path = '/teamspace/studios/this_studio/wavelet_train_data.csv'
categories = list(pd.read_csv(train_path)['label'].unique())
from torch.utils.data import Dataset


import pandas as pd
from PIL import Image
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        image = Image.open(img_name).convert("RGB")
        label_str = self.data.iloc[idx, 1]
        label = categories.index(label_str)

        return image, label

def collate_fn(batch):
    images, label = zip(*batch)
    inputs = feature_extractor(images=list(images), return_tensors='pt').pixel_values.to(device)
    label = torch.tensor(label).to(device)
    return inputs, label


from torch.utils.data import DataLoader

# Create DataLoader for training and validation sets
dataset = DiabeticRetinopathyDataset(csv_file=train_path)
train_loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)
dataset = DiabeticRetinopathyDataset(csv_file=train_path)
val_loader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)

import torch.nn as nn
        
num_classes = len(categories)

from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(categories),
    id2label={str(i): c for i, c in enumerate(categories)},
    label2id={c: str(i) for i, c in enumerate(categories)}
)

import os

# model_path = '/teamspace/studios/this_studio/vit_finetuned_original.pth'
# if os.path.exists(model_path):
#     print('Loading the Pretrained model')
#     model.load_state_dict(torch.load(model_path))

model.to(device)
import torch.optim as optim

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=2e-5)


from tqdm import tqdm

# Number of epochs for training
num_epochs = 50
with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
    f.write('\nVIT\n')
# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize running loss for the current epoch
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: 0.0000")  # Initialize progress bar
    with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
            f.write(f'\nEpoch {epoch+1}/{num_epochs}\n')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")  # Update progress bar with current loss

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')  # Print average loss for the epoch
    with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
            f.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}\n')

    # Validation
    # model.eval()  # Set the model to evaluation mode
    # correct = 0  # Initialize correct predictions counter
    # total = 0  # Initialize total samples counter
    
    # with torch.no_grad():  # Disable gradient calculation for validation
    #     for images, labels in val_loader:
    #         images, labels = images.to(device), labels.to(device)  # Move images and labels to the device
    #         outputs = model(images)  # Forward pass: compute predicted outputs by passing inputs to the model
    #         _, predicted = torch.max(outputs.logits, 1)  # Get the class label with the highest probability
    #         total += labels.size(0)  # Update total samples
    #         correct += (predicted == labels).sum().item()  # Update correct predictions

    # print(f'Validation Accuracy: {100 * correct / total}%')
    # with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
    #     f.write(f'Validation Accuracy: {100 * correct / total}%\n')

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'vit_finetuned_wavelet.pth')

# python /teamspace/studios/this_studio/diabetic_retinopathy/vitt.py
# python /teamspace/studios/this_studio/diabetic_retinopathy/vitt_cat.py
