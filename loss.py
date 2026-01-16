#time: 2026-1-15 15:57
import torch
from torch import nn
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.location_loss = nn.MSELoss()
        self.classes_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        outputs_location = outputs[:, 0:4]
        outputs_classes = outputs[:, 4:]  # 仍然是4维logits
        targets_location = targets[:, 0:4]
        # 直接取类别索引，转为long类型
        targets_classes_idx = targets[:, 4].long()
        loss = self.location_loss(outputs_location, targets_location) + self.classes_loss(outputs_classes, targets_classes_idx)
        return loss
    

if __name__ == "__main__":
    criterion = Loss()
