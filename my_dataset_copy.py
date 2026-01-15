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

        label_path = os.path.join(self.label_dir, os.path.splitext(self.imgs[idx])[0] + ".xml") #already aligned
        with open(label_path, 'r', encoding='utf-8') as f:
            label_content = f.read()
        label_dict = xmltodict.parse(label_content)
        objects = label_dict['annotation']['object']
        targets = []
        for obj in objects:
            name = obj['name']
            id_ = self.classes.index(name)
            bndbox = obj['bndbox']
            xmin = float(bndbox['xmin'])
            ymin = float(bndbox['ymin'])
            xmax = float(bndbox['xmax'])
            ymax = float(bndbox['ymax'])
            targets.extend([id_, xmin, ymin, xmax, ymax])
        targets = torch.tensor(targets, dtype=torch.float32)
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
