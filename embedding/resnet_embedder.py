import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

class ResNetEmbedder:
    def __init__(self, device="cpu"):
        self.device = device

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity() 
        model.eval()

        self.model = model.to(device)

        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    @torch.no_grad()
    def embed(self, img_path):
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        feat = self.model(x).squeeze(0)
        feat = feat / feat.norm(p=2)  # L2 normalize
        return feat.cpu()
