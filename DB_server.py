from flask import Flask,send_file
from flask import request,jsonify
from flask_cors import CORS
import io
import numpy as np
import cv2
from PIL import Image

from torchvision import models, transforms
import torch
from PIL import Image
import torch.nn as nn


app=Flask(__name__)
CORS(app)

model_path = '/kaggle/working/resnet_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load(model_path, map_location = device))

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

def predict(image):
    sigmaX = 10
    gaussian = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4, 128)
    image = cv2.resize(gaussian, (224, 224))

    image = Image.fromarray(image)

    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

    categories = ['Mild', 'Proliferate', 'Moderate', 'No DR', 'Severe']
    return categories[predicted]



@app.route("/", methods=['POST'])
def testing():
    data = request.get_json()
    text = data.get('text', '')
    print("text",text)
    return jsonify({'text': text})


@app.route("/members", methods=['POST','GET'])
def members():
    if request.method == 'GET':
        return "Send an image, name, and age using POST method"

    data = request.form
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    name = data.get('name')
    age = data.get('age')

    image = Image.open(io.BytesIO(file.read()))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    prediction = predict(image)
    print('PREDICTION COMPLETE')

    return jsonify({
        "Output Generated": prediction,
        "name": name,
        "age": age
    })
    

    

if __name__=="__main__":
    app.run(debug=True)

