"""
Modified from https://github.com/OSU-NLP-Group/MagicBrush/blob/ecd0b07560bb26381dbb9ae7ea69acde96e06cc7/evaluation/image_eval.py
"""

from torchvision.transforms import transforms
from PIL import Image
from torch import nn

def eval_distance(src_image_path,tgt_image_path, metric='l1'):
    """
    Using pytorch to evaluate l1 or l2 distance
    """
    if metric == 'l1':
        criterion = nn.L1Loss()
    elif metric == 'l2':
        criterion = nn.MSELoss()
    eval_score = 0
    img_0 = Image.open(src_image_path).convert('RGB')
    img_1 = Image.open(tgt_image_path).convert('RGB')
    # resize to gt size
    img_0 = img_0.resize(img_1.size)
    # convert to tensor
    img_0 = transforms.ToTensor()(img_0)
    img_1 = transforms.ToTensor()(img_1)
    # calculate distance
    return criterion(img_0, img_1).detach().cpu().numpy().item()
