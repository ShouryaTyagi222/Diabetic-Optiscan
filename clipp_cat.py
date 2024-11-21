import clip
import torch

# OpenAI CLIP model and preprocessing
model, preprocess = clip.load("ViT-B/32", jit=False)
print()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# model_ft = torch.load('clip_finetuned_entire_model.pth')
# model_ft.to(device)
import pandas as pd
subcategories = list(pd.read_csv('/teamspace/studios/this_studio/diabetic_retinopathy/train_data.csv')['label'].unique())
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
        img_name = self.data.iloc[idx, 1]
        image = Image.open(img_name).convert("RGB")
        label_str = self.data.iloc[idx, 2]
        label = subcategories.index(label_str)

        if self.transform:
            image = self.transform(image)

        return image, label


from torch.utils.data import DataLoader

# Create DataLoader for training and validation sets
dataset = DiabeticRetinopathyDataset(csv_file='/teamspace/studios/this_studio/diabetic_retinopathy/train_data.csv',
                                     transform=preprocess)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
dataset = DiabeticRetinopathyDataset(csv_file='/teamspace/studios/this_studio/diabetic_retinopathy/train_data.csv',
                                     transform=preprocess)
val_loader = DataLoader(dataset, batch_size=128, shuffle=False)

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
model_ft.load_state_dict(torch.load('/teamspace/studios/this_studio/clip_finetuned_wavelet.pth', map_location=device))


import torch.optim as optim

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)


from tqdm import tqdm

# Number of epochs for training
num_epochs = 100
with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
    f.write('\nCLIP\n')

from sklearn.metrics import classification_report
from tqdm import tqdm

# Validation
model_ft.eval()  # Set the model to evaluation mode
all_labels = []  # List to store true labels
all_predictions = []  # List to store predicted labels

# Initialize tqdm progress bar for validation
pbar = tqdm(val_loader, desc="Validation", total=len(val_loader))

with torch.no_grad():  # Disable gradient calculation for validation
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)  # Move images and labels to the device
        outputs = model_ft(images)  # Forward pass: compute predicted outputs by passing inputs to the model
        _, predicted = torch.max(outputs.data, 1)  # Get the class label with the highest probability
        
        all_labels.extend(labels.cpu().numpy())  # Convert to CPU and add true labels to the list
        all_predictions.extend(predicted.cpu().numpy())  # Convert to CPU and add predictions to the list

        # Update progress bar description with current progress (optional)
        pbar.set_postfix({"Processed": len(all_labels)})

# Generate classification report
report = classification_report(all_labels, all_predictions, target_names=subcategories)

# Print or save the report
print(report)

# Optionally, write to a log file
with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
    f.write(f'Classification Report:\n{report}\n')


# python /teamspace/studios/this_studio/diabetic_retinopathy/clipp.py
# python /teamspace/studios/this_studio/diabetic_retinopathy/clipp_cat.py
