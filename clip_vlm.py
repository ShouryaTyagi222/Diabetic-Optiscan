import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import pandas as pd
import os
from PIL import Image
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
            'Proliferate': 4  # Map string labels to integers
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 1]
        image = Image.open(img_name).convert("RGB")
        label_str = self.data.iloc[idx, 2]
        label = self.class_to_idx[label_str]  # Convert string label to index

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Convert image to tensor with values in [0, 1]
])



dataset = DiabeticRetinopathyDataset(csv_file='/teamspace/studios/this_studio/diabetic_retinopathy/train_data.csv',
                                     transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

dataset = DiabeticRetinopathyDataset(csv_file='/teamspace/studios/this_studio/diabetic_retinopathy/train_data.csv',
                                     transform=transform)

test_loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

from tqdm import tqdm

def evaluate_clip_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            text_inputs = [f"This is a {label.item()}" for label in labels]
            inputs = processor(text=text_inputs, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image

            _, predicted = torch.max(logits_per_image, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
            f.write(f'Accuracy: {accuracy * 100:.2f}% \n\n')

# Evaluate the model



num_epochs = 1
for epoch in range(num_epochs):
    total_loss = 0
    # Wrap the DataLoader with tqdm for batch processing visualization
    model.train()
    with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
            f.write(f'Epoch {epoch + 1}/{num_epochs} \n')
    
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
        for images, labels in train_loader:
            optimizer.zero_grad()

            # Move data to the correct device
            images = images.to(device)
            labels = labels.to(device)  # Ensure labels are on the same device

            # Prepare the text inputs (e.g., 'This is a 0', 'This is a 1', etc.)
            text_inputs = [f"This is a {label.item()}" for label in labels]
            inputs = processor(text=text_inputs, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            # Calculate loss
            loss = (torch.nn.functional.cross_entropy(logits_per_image, labels) +
                    torch.nn.functional.cross_entropy(logits_per_text, labels)) / 2
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())

        # Calculate and print average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
            f.write(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f} \n')

        evaluate_clip_model(model, test_loader)

    torch.save(model.state_dict(), 'clip_diabetic_retinopathy_model.pth')