import torch
from torchvision.transforms import transforms
from PIL import Image
import torch.nn.functional as F

class Dinov2Similarity:
    def __init__(self, model_name="dinov2_vitb14", device="cuda:1"):
        self.model=torch.hub.load('facebookresearch/dinov2', model_name)
        self.device=device
        self.model.eval()
        self.model.to(self.device)
        print(f"Dino v2 {model_name} loaded successfully to device {self.device}!!!")

    def encode_image(self, img_path):
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        img=Image.open(img_path).convert('RGB')
        img=transform(img).unsqueeze(0).to(self.device)

        return self.model(img).detach().cpu().float()

    def get_similarity(self, src_img_path, tgt_img_path):
        src_img_features=self.encode_image(src_img_path)
        tgt_img_features = self.encode_image(tgt_img_path)

        return F.cosine_similarity(src_img_features, tgt_img_features).item()
