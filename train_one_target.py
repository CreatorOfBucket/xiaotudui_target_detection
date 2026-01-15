from model import mynetwork
from torch.utils.data import DataLoader
import torch
from dataset import mydata
from torchvision import transforms
from loss import Loss

num_epochs = 10
batch_size = 8

train_img_dir = r"E:\HelmetDataset-VOC\train\images"
train_label_dir = r"E:\HelmetDataset-VOC\train\labels"

# 指定要训练的类别
TARGET_CLASS = "with helmet"
CLASSES = ["no helmet", "motor", "number", "with helmet"]
TARGET_CLASS_ID = CLASSES.index(TARGET_CLASS)

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

train_dataset = mydata(
    img_dir=train_img_dir,
    label_dir=train_label_dir,
    transform=transform,
)


def collate_fn(batch):
    """只保留指定类别的目标，过滤掉没有该类别的图片"""
    images = []
    targets = []
    for img, t in batch:
        if t.numel() == 0:
            continue
        # 找出属于目标类别的目标（one-hot 位置为 4 + TARGET_CLASS_ID）
        mask = t[:, 4 + TARGET_CLASS_ID] == 1.0
        filtered = t[mask]
        # 空检查，防止没有该类别的目标
        if filtered.numel() == 0:
            continue
        # 只取第一个匹配的目标
        images.append(img)
        targets.append(filtered[0])
    
    if len(images) == 0:
        return None, None
    
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = mynetwork().to(device)
loss_fn = Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for images, targets in train_loader:
        # 跳过空 batch（没有目标类别的图片）
        if images is None:
            continue
        
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")
