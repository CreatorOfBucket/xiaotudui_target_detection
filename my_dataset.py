'''
Created on 2026.1.14
看着视频敲了一遍
再做一个copy自己完全敲
'''
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import xmltodict
import torch
class VOCdata(Dataset):
    def __init__(self,img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.imgs = os.listdir(self.img_dir)
        self.labels = os.listdir(self.label_dir)
        self.classes = ["no helmet", "motor", "number", "with helmet"]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        label_path = os.path.join(self.label_dir, self.labels[idx])
        with open(label_path, 'r',encoding='utf-8') as f:
            label_content = f.read()
        label_dict = xmltodict.parse(label_content)
        objects = label_dict['annotation']['object']
        targets = []
        for obj in objects:#找obj，从debug里一步一步找
            name = obj['name']
            id_ = self.classes.index(name)
            bndbox = obj['bndbox']
            xmin = float(bndbox['xmin'])
            ymin = float(bndbox['ymin'])
            xmax = float(bndbox['xmax'])
            ymax = float(bndbox['ymax'])
            targets.extend([id_, xmin, ymin, xmax, ymax])#直接拼接
        targets = torch.tensor(targets, dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, targets
    

if __name__ == "__main__":
    dataset = VOCdata(img_dir=r"E:\HelmetDataset-VOC\train\images", label_dir=r"E:\HelmetDataset-VOC\train\labels",transform=transforms.ToTensor())
    print(f"Dataset size: {len(dataset)}")
    img, targets = dataset[0]
    print(f"image: {img}")
    print(f"targets: {targets}")
    print(type(targets))