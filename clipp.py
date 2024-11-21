import clip
import torch

# OpenAI CLIP model and preprocessing
model, preprocess = clip.load("ViT-B/32", jit=False)
print()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# model_ft = torch.load('clip_finetuned_entire_model.pth')
train_path = '/teamspace/studios/this_studio/wavelet_train_data.csv'
# model_ft.to(device)
import pandas as pd
subcategories = list(pd.read_csv(train_path)['label'].unique())
from torchvision import transforms
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
        label = subcategories.index(label_str)

        if self.transform:
            image = self.transform(image)

        return image, label


from torch.utils.data import DataLoader

# Create DataLoader for training and validation sets
dataset = DiabeticRetinopathyDataset(csv_file=train_path,
                                     transform=preprocess)
train_loader = DataLoader(dataset, batch_size=64)
dataset = DiabeticRetinopathyDataset(csv_file=train_path,
                                     transform=preprocess)
val_loader = DataLoader(dataset, batch_size=128)

import torch.nn as nn

# Modify the model to include a classifier for subcategories
class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()  # Convert to float32
        return self.classifier(features)
        
num_classes = len(subcategories)
model_ft = CLIPFineTuner(model, num_classes).to(device)
model_path = ''
# model_ft.load_state_dict(torch.load(model_path, map_location=device))


import torch.optim as optim

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)

from tqdm import tqdm

# Number of epochs for training
num_epochs = 150
with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
    f.write('\nCLIP\n')
# Training loop
for epoch in range(num_epochs):
    model_ft.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize running loss for the current epoch
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: 0.0000")  # Initialize progress bar
    with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
            f.write(f'\nEpoch {epoch+1}/{num_epochs}\n')
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_ft(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")  # Update progress bar with current loss

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')  # Print average loss for the epoch
    with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
            f.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}\n')

    # Validation
    # model_ft.eval()  # Set the model to evaluation mode
    # correct = 0  # Initialize correct predictions counter
    # total = 0  # Initialize total samples counter
    
    # with torch.no_grad():  # Disable gradient calculation for validation
    #     for images, labels in val_loader:
    #         images, labels = images.to(device), labels.to(device)  # Move images and labels to the device
    #         outputs = model_ft(images)  # Forward pass: compute predicted outputs by passing inputs to the model
    #         _, predicted = torch.max(outputs.data, 1)  # Get the class label with the highest probability
    #         total += labels.size(0)  # Update total samples
    #         correct += (predicted == labels).sum().item()  # Update correct predictions

    # print(f'Validation Accuracy: {100 * correct / total}%')
    # with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
    #     f.write(f'Validation Accuracy: {100 * correct / total}%\n')

# Save the fine-tuned model
torch.save(model_ft.state_dict(), 'clip_finetuned_wavelet.pth')