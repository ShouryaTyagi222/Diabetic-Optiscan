import torch

from transformers import ViTFeatureExtractor

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd
categories = list(pd.read_csv('/teamspace/studios/this_studio/diabetic_retinopathy/train_data.csv')['label'].unique())
from torch.utils.data import Dataset

model_path = '/teamspace/studios/this_studio/vit_finetuned_wavelet.pth'


import pandas as pd
from PIL import Image
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 1]
        image = Image.open(img_name).convert("RGB")
        label_str = self.data.iloc[idx, 2]
        label = categories.index(label_str)

        return image, label

def collate_fn(batch):
    images, label = zip(*batch)
    inputs = feature_extractor(images=list(images), return_tensors='pt').pixel_values.to(device)
    label = torch.tensor(label).to(device)
    return inputs, label


from torch.utils.data import DataLoader

# Create DataLoader for training and validation sets
dataset = DiabeticRetinopathyDataset(csv_file='/teamspace/studios/this_studio/diabetic_retinopathy/train_data.csv',)
train_loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
dataset = DiabeticRetinopathyDataset(csv_file='/teamspace/studios/this_studio/diabetic_retinopathy/train_data.csv')
val_loader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn, shuffle=False)

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
if os.path.exists(model_path):
    print('Loading the Pretrained model')
    model.load_state_dict(torch.load(model_path))

model.to(device)
import torch.optim as optim

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=2e-5)


from sklearn.metrics import classification_report
from tqdm import tqdm

# Validation
model.eval()  # Set the model to evaluation mode
all_labels = []  # List to store true labels
all_predictions = []  # List to store predicted labels

# Initialize tqdm progress bar for validation
pbar = tqdm(val_loader, desc="Validation", total=len(val_loader))

with torch.no_grad():  # Disable gradient calculation for validation
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)  # Move images and labels to the device
        outputs = model(images)  # Forward pass: compute predicted outputs by passing inputs to the model
        _, predicted = torch.max(outputs.logits, 1)  # Get the class label with the highest probability
        
        all_labels.extend(labels.cpu().numpy())  # Convert to CPU and add true labels to the list
        all_predictions.extend(predicted.cpu().numpy())  # Convert to CPU and add predictions to the list

        # Update progress bar description with current progress (optional)
        pbar.set_postfix({"Processed": len(all_labels)})

# Generate classification report
report = classification_report(all_labels, all_predictions, target_names=categories)

# Print or save the report
print(report)

# Optionally, write to a log file
with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
    f.write(f'Classification Report:\n{report}\n')



