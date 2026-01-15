#created on 2026.1.14
from PIL import Image
from torch.utils.data import Dataset
import os
import xmltodict
import torch
from torchvision import transforms

class mydata(Dataset):
    def __init__(self,img_dir,label_dir,transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.imgs = sorted(os.listdir(self.img_dir))
        # self.labels = sorted(os.listdir(self.label_dir)) #没有对齐，需要一一对应
        self.classes = ["no helmet", "motor", "number", "with helmet"]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
         
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size  # 获取原图尺寸用于归一化

        label_path = os.path.join(self.label_dir, os.path.splitext(self.imgs[idx])[0] + ".xml") #already aligned
        with open(label_path, 'r', encoding='utf-8') as f:
            label_content = f.read()
        label_dict = xmltodict.parse(label_content)
        objects = label_dict['annotation']['object']
        targets = []
        num_classes = len(self.classes)
        for obj in objects:
            name = obj['name']
            id_ = self.classes.index(name)
            bndbox = obj['bndbox']
            xmin = float(bndbox['xmin']) / orig_w  # 归一化到 [0, 1]
            ymin = float(bndbox['ymin']) / orig_h
            xmax = float(bndbox['xmax']) / orig_w
            ymax = float(bndbox['ymax']) / orig_h
            target_vec = torch.zeros(4 + num_classes, dtype=torch.float32)
            target_vec[0:4] = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
            target_vec[4 + id_] = 1.0
            targets.append(target_vec)
        targets = torch.stack(targets) if targets else torch.zeros((0, 4 + num_classes), dtype=torch.float32)
        if self.transform:
            img = self.transform(img)

        return img, targets

if __name__ == "__main__":
    dataset = mydata(img_dir=r"E:\HelmetDataset-VOC\train\images", label_dir=r"E:\HelmetDataset-VOC\train\labels", transform=transforms.ToTensor())
    print(f"Dataset size: {len(dataset)}")
    img, targets = dataset[10]
    print(f"image: {img}")
    print(f"targets: {targets}")
    print(type(targets))
