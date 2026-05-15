import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# class labels
classes = ["healthy","multiple_diseases","rust","scab"]

# CNN model (same architecture you trained)
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*32,128),
            nn.ReLU(),
            nn.Linear(128,4)
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# load trained model
model = CNN()
model.load_state_dict(torch.load("plant_model.pth", map_location=torch.device("cpu")))
model.eval()

# image transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Streamlit UI
st.title("🌿 Plant Disease Detection AI")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Leaf", use_container_width=True)

    input_img = transform(img).unsqueeze(0)

    with torch.no_grad():

        output = model(input_img)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(probabilities,1)

    disease = classes[predicted]

    st.subheader(f"Prediction: {disease}")
    st.write(f"Confidence: {confidence.item()*100:.2f}%")