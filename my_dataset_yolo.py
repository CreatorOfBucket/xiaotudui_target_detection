import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch

class YOLOdata(Dataset):
    def __init__(self,img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.imgs = sorted(os.listdir(self.img_dir))

        self.transform = transform

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        label_path = os.path.join(self.label_dir, os.path.splitext(self.imgs[idx])[0] + ".txt")
        targets = []
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = float(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    targets.extend([cls_id, x_c, y_c, w, h])
        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0,5), dtype=torch.float32)
        
        
        if self.transform:
            img = self.transform(img)
        return img, targets
    

if __name__ == "__main__":
    dataset = YOLOdata(img_dir=r"E:\HelmetDataset-YOLO-Train\images", label_dir=r"E:\HelmetDataset-YOLO-Train\labels",transform=transforms.ToTensor())
    print(f"Dataset size: {len(dataset)}")
    img, targets = dataset[0]
    print(f"image: {img}")
    print(f"targets: {targets}")
    print(type(targets))