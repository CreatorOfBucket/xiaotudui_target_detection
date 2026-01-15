from model import mynetwork
from torch.utils.data import DataLoader
import torch
from dataset import mydata
from torchvision import transforms
from loss import Loss

num_epochs = 10

train_img_dir = r"E:\HelmetDataset-VOC\train\images"
train_label_dir = r"E:\HelmetDataset-VOC\train\labels"

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
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
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
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')