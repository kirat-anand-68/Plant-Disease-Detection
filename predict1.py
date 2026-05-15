import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# class names
classes = ["healthy","multiple_diseases","rust","scab"]

# same model structure as training
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


# image transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


# load image
img = Image.open("dataset/images/Train_5.jpg").convert("RGB")

img = transform(img)

img = img.unsqueeze(0)

# prediction
output = model(img)

prediction = torch.argmax(output)

print("Predicted Disease:", classes[prediction])