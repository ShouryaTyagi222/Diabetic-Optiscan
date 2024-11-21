from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and processor
model_path = "microsoft/Florence-2-base-ft"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code = True).to(device)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code = True)

# Load test data
test_data_path = '/teamspace/studios/this_studio/diabetic_retinopathy/train_data.csv'  # Replace with your test CSV path

# Ensure CUDA is available and empty cache
torch.cuda.empty_cache()

# Define a custom Dataset class
class RetinopathyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.processor = processor
        self.question = "What is the diabetic retinopathy severity in this retinal image?"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image path and label
        image_id = self.data.iloc[idx, 1]
        answer = self.data.iloc[idx, 2]

        # Load the image
        image = Image.open(image_id).convert("RGB").resize((64,64))
        text = self.question

        return text, answer, image

def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers
    

# Create DataLoader
batch_size = 8  # Adjust based on your GPU's memory capacity
test_dataset = RetinopathyDataset(test_data_path)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Model inference
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, true_labels in tqdm(test_dataloader, desc="Processing Batches"):
        outputs = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_length=50)
        
        # Decode predictions
        predictions = [processor.tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
        
        # Append results
        y_true.extend(true_labels)
        y_pred.extend(predictions)
        print('true labels :', true_labels)
        print('prediction :', predictions)

# Generate and print the classification report
print("Classification Report:")
print(classification_report(y_true, y_pred))
