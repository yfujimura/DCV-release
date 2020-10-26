import torch
from torch import nn
import torch.nn.functional as F

class AirlightEstimator(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.conv1 = self._convLayer(3, 16, 7)
        self.conv2 = self._convLayer(16, 32, 5)
        self.conv3 = self._convLayer(32, 64, 3)
        self.conv4 = self._convLayer(64, 128, 3)
        self.conv5 = self._convLayer(128, 256, 3)
        self.conv6 = self._convLayer(256, 256, 3)
        self.conv7 = self._convLayer(256, 64, 1, stride=1)
        self.conv8 = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        
    def forward(self, reference_images):
        # x : N x 3 x 192 x 256
        
        x = self.conv1(reference_images) # N x 16 x 96 x 128
        x = self.conv2(x) # N x 32 x 48 x 64
        x = self.conv3(x) # N x 64 x 24 x 32
        x = self.conv4(x) # N x 128 x 12 x 16
        x = self.conv5(x) # N x 256 x 6 x 8
        x = self.conv6(x) # N x 256 x 3 x 4
        x = F.adaptive_avg_pool2d(x, (1,1)) # N x 256 x 1 x 1
        x = self.conv7(x) # N x 64 x 1 x 1
        A = self.conv8(x) # N x 1 x 1 x 1
        
        return A.squeeze(3).squeeze(2) # N x 1
        
        
    def _convLayer(self, in_channels, out_channels, kernel_size, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )