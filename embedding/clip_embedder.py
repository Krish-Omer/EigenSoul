import torch
import clip
from PIL import Image

class CLIPEmbedder:
    def __init__(self, device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, img_path):
        img = Image.open(img_path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x).squeeze(0)
        feat = feat / feat.norm(p=2)  # L2 normalize
        return feat.cpu()
