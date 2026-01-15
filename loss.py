#time: 2026-1-15 15:57
import torch
from torch import nn
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.location_loss = nn.MSELoss()
        self.classes_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        outputs_location = outputs[:,0:4]
        outputs_classes = outputs[:,4:]
        targets_location = targets[:,0:4]
        targets_classes = targets[:,4:]
        loss = self.location_loss(outputs_location, targets_location) + self.classes_loss(outputs_classes, targets_classes)
        return loss
    

if __name__ == "__main__":
    criterion = Loss()
    