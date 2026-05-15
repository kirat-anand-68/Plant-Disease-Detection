import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

classes = ["healthy","multiple_diseases","rust","scab"]

# CNN architecture (same as training)
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


# load model
model = CNN()
model.load_state_dict(torch.load("plant_model.pth"))
model.eval()


transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


# choose image
image_path = "dataset/images/Train_50.jpg"

img = Image.open(image_path).convert("RGB")

input_img = transform(img).unsqueeze(0)

# prediction
with torch.no_grad():

    outputs = model(input_img)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities,1)

disease = classes[predicted]

print("Prediction:",disease)
print("Confidence:",round(confidence.item()*100,2),"%")

# show image
plt.imshow(img)
plt.title(f"{disease} ({confidence.item()*100:.2f}%)")
plt.axis("off")

plt.savefig("prediction.png")

print("Prediction image saved as prediction.png")