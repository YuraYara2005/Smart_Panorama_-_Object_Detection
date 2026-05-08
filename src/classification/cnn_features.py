import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
device = torch.device("cpu")

model = models.mobilenet_v2(pretrained=True)

model.classifier = torch.nn.Identity()

model.eval()

model.to(device)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_cnn_features(image, bbox):

    x, y, w, h = bbox

    crop = image[y:y+h, x:x+w]

    if crop.size == 0:
        return None

    crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    tensor = transform(crop)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():
        features = model(tensor)
    return features.cpu().numpy().flatten()