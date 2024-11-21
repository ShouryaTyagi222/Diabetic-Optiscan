from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model_path = 'E:/diabetic_rectinography/resnet_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load(model_path, map_location=device))

model = model.to(device)
model.eval()

# Transformations for the input image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def predict(image):
    sigmaX = 10
    gaussian = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    image = cv2.resize(gaussian, (224, 224))

    image = Image.fromarray(image)

    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

    categories = ['Mild', 'Proliferate', 'Moderate', 'No DR', 'Severe']
    return categories[predicted]

@app.get("/")
async def testing():
    return JSONResponse("SEND IMAGE, NAME AND AGE")


@app.post("/members")
async def members(file: UploadFile = File(...) ,gender : str = Form(...), other_diseases: str = Form(...), age: str = Form(...), pregnant: str = Form(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        prediction = predict(image)
        return JSONResponse(content={"op" : f'The Person Aged {age} {gender} having {other_diseases} who is also {pregnant} suffers from {prediction} diabetic retinopathy disease'})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Running the app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
