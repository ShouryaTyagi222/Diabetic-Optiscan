from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import os
from PIL import Image
import pandas as pd
# from torch import utils
import gc
from transformers import get_scheduler
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

epochs = 1
batch_size = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base-ft",
    trust_remote_code=True,
    revision='refs/pr/6'
).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft",
    trust_remote_code=True, revision='refs/pr/6')

torch.cuda.empty_cache()
torch.manual_seed(42)

class VQADataset(torch.utils.data.Dataset):
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

dataset = VQADataset(csv_file='/teamspace/studios/this_studio/diabetic_retinopathy/train_data.csv',)

train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)


optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
num_training_steps = epochs * len(train_dataloader)

lr_scheduler = get_scheduler(name="linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_training_steps,)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    i = -1
    with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
            f.write(f'epoch: {epoch}/{epochs} \n')
    train_progress_bar = tqdm(range(len(train_dataloader)), desc='Training batch: ...')

    for idx, batch in zip(train_progress_bar, train_dataloader):
        i += 1
        inputs, answers = batch

        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs["pixel_values"].to(device)
        labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        # outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        del inputs, answers, input_ids, pixel_values, labels
        gc.collect()
        torch.cuda.empty_cache()

        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        train_loss += loss.item()
        train_progress_bar.set_postfix({'Batch Loss': loss.item()})

    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Average Training Loss: {avg_train_loss}")
    output_dir = '/teamspace/studios/this_studio/diabetic_retinopathy/florence_model'
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    with open('/teamspace/studios/this_studio/diabetic_retinopathy/log.txt', 'a') as f:
            f.write(f'Average Training Loss: {avg_train_loss} \n\n')
processor.save_pretrained(output_dir)