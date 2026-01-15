#time: 2026-1-15 15:57
from torch import nn
import torch
import torchvision


class mynetwork(nn.Module):
    def __init__(self):
        super(mynetwork, self).__init__()
        
        self.model = torchvision.models.vgg16(weights=None).features
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 8)
        )


    def forward(self, x):
        
        x = self.model(x)
        x = self.pool(x)
        x = self.fc_layers(x)
        return x
    

if __name__ == "__main__":
    net = mynetwork().cuda()
    input_tensor = torch.randn(1, 3, 448, 448).cuda()
    output = net(input_tensor)
    print(net)
    print("Output shape:", output.shape)
    torch.onnx.export(
        net,
        input_tensor,
        "mynetwork.onnx",
        opset_version=11,
        use_external_data_format=True
    )


