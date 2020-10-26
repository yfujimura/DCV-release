import torch
from torch import nn
import torch.nn.functional as F


class MVSNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = self._downConvLayer(259, 128, 7)
        self.conv2 = self._downConvLayer(128, 256, 5)
        self.conv3 = self._downConvLayer(256, 512, 3)
        self.conv4 = self._downConvLayer(512, 512, 3)
        self.conv5 = self._downConvLayer(512, 512, 3)
        
        self.upconv4 = self._upConvLayer(512, 512, 3)
        self.iconv4 = self._convLayer(1024, 512, 3)
        
        self.upconv3 = self._upConvLayer(512, 512, 3)
        self.iconv3 = self._convLayer(1024, 512, 3)
        self.disp3 = self._dispLayer(512)
        
        self.upconv2 = self._upConvLayer(512, 256, 3)
        self.iconv2 = self._convLayer(513, 256, 3)
        self.disp2 = self._dispLayer(256)
        
        self.upconv1 = self._upConvLayer(256, 128, 3)
        self.iconv1 = self._convLayer(257, 128, 3)
        self.disp1 = self._dispLayer(128)
        
        self.upconv0 = self._upConvLayer(128, 64, 3)
        self.iconv0 = self._convLayer(65, 64, 3)
        self.disp0 = self._dispLayer(64)
        
    def forward(self, reference_images, source_images, KRKiUV, Kt):
        cost_volume = self._computeCostVolume(reference_images, source_images, KRKiUV, Kt)
        
        # reference_images: samples x 3 x height x width
        # cost_volume: samples x depth_samples x height x width 
        
        scale = 2.0
        
        x = torch.cat((reference_images, cost_volume), 1)  # height x width
        
        conv1 = self.conv1(x)  # height /2 x width /2  (96, 128)
        conv2 = self.conv2(conv1)  # height /4 x width /4  (48, 64)
        conv3 = self.conv3(conv2)  # height /8 x width /8  (24, 32)
        conv4 = self.conv4(conv3)  # height /16 x width /16  (12, 16)
        conv5 = self.conv5(conv4)  # height /32 x width /32   (6, 8)
        
        upconv4 = self.upconv4(conv5)  # height /16 x width /16   
        iconv4 = self.iconv4(torch.cat((upconv4, conv4), 1))
        
        upconv3 = self.upconv3(iconv4)  # height /8 x width /8
        iconv3 = self.iconv3(torch.cat((upconv3, conv3), 1))
        disp3 = scale * self.disp3(iconv3) # height /8 x width /8
        udisp3 = F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=True) # height /4 x width /4
        
        upconv2 = self.upconv2(iconv3)  # height /4 x width /4
        iconv2 = self.iconv2(torch.cat((upconv2, conv2, udisp3), 1))
        disp2 = scale * self.disp2(iconv2)
        udisp2 = F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=True)
        
        upconv1 = self.upconv1(iconv2)  # height /2 x width /2
        iconv1 = self.iconv1(torch.cat((upconv1, conv1, udisp2), 1))
        disp1 = scale * self.disp1(iconv1)
        udisp1 = F.interpolate(disp1, scale_factor=2, mode='bilinear', align_corners=True)
        
        upconv0 = self.upconv0(iconv1)  # height x width
        iconv0 = self.iconv0(torch.cat((upconv0, udisp1), 1))
        disp0 = scale * self.disp0(iconv0)
        
        return [disp0, disp1, disp2, disp3]
        
        
    def _downConvLayer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _upConvLayer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _convLayer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _dispLayer(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def _computeCostVolume(self, reference_images, source_images, KRKiUV, Kt):
        # Input:
        #   images: samples x 3 x height x width 
        #   KRKiUV: samples x 3 x pixels
        #   Kt: samples x 3 
        # Output:
        #   cost_volume: samples x depth_samples x height x width
        
        depth_max = 50.0
        depth_min = 0.5
        steps = 256
        idepth_min = 1.0 / depth_max
        idepth_max = 1.0 / depth_min
        idepth_step = (idepth_max - idepth_min) / (steps-1)
        
        cost_volume = torch.zeros(reference_images.shape[0], steps, reference_images.shape[2], reference_images.shape[3]).to('cuda:0')
        
        for i in range(steps):
            depth = 1.0 / (idepth_min + i * idepth_step)
            warped = self._warpImage(source_images, depth, KRKiUV, Kt)
            abs_residual = torch.sum(torch.abs(reference_images - warped), dim=1)
            cost_volume[:,i,:,:] = abs_residual
        
        return cost_volume
    
    def _warpImage(self, source_images, depth, KRKiUV, Kt):
        batch_n = source_images.shape[0]
        width = source_images.shape[3]
        height = source_images.shape[2]
        
        normalize_base = torch.FloatTensor([width/2, height/2]).to('cuda')
        normalize_base = normalize_base.unsqueeze(0).unsqueeze(-1)
        
        warped_uv = depth * KRKiUV + Kt.unsqueeze(-1) # samples x 3 x pixels
        demon = warped_uv[:,2,:].unsqueeze(1)
        warped_uv = warped_uv[:,0:2,:] / (demon + 1e-6)
        warped_uv = (warped_uv - normalize_base) / normalize_base
        warped_uv = warped_uv.view(batch_n, 2, width, height)
        warped_uv = warped_uv.transpose(1,3)
        warped = F.grid_sample(source_images, warped_uv)
        
        
        return warped
    
class DehazingMVSNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = self._downConvLayer(259, 128, 7)
        self.conv2 = self._downConvLayer(128, 256, 5)
        self.conv3 = self._downConvLayer(256, 512, 3)
        self.conv4 = self._downConvLayer(512, 512, 3)
        self.conv5 = self._downConvLayer(512, 512, 3)
        
        self.upconv4 = self._upConvLayer(512, 512, 3)
        self.iconv4 = self._convLayer(1024, 512, 3)
        
        self.upconv3 = self._upConvLayer(512, 512, 3)
        self.iconv3 = self._convLayer(1024, 512, 3)
        self.disp3 = self._dispLayer(512)
        
        self.upconv2 = self._upConvLayer(512, 256, 3)
        self.iconv2 = self._convLayer(513, 256, 3)
        self.disp2 = self._dispLayer(256)
        
        self.upconv1 = self._upConvLayer(256, 128, 3)
        self.iconv1 = self._convLayer(257, 128, 3)
        self.disp1 = self._dispLayer(128)
        
        self.upconv0 = self._upConvLayer(128, 64, 3)
        self.iconv0 = self._convLayer(65, 64, 3)
        self.disp0 = self._dispLayer(64)
        
    def forward(self, reference_images, source_images, KRKiUV, Kt, A, beta):
        cost_volume = self._computeDehazingCostVolume(reference_images, source_images, KRKiUV, Kt, A, beta)
        
        # reference_images: samples x 3 x height x width
        # cost_volume: samples x depth_samples x height x width 
        
        scale = 2.0
        
        x = torch.cat((reference_images, cost_volume), 1)  # height x width
        
        conv1 = self.conv1(x)  # height /2 x width /2  (96, 128)
        conv2 = self.conv2(conv1)  # height /4 x width /4  (48, 64)
        conv3 = self.conv3(conv2)  # height /8 x width /8  (24, 32)
        conv4 = self.conv4(conv3)  # height /16 x width /16  (12, 16)
        conv5 = self.conv5(conv4)  # height /32 x width /32   (6, 8)
        
        upconv4 = self.upconv4(conv5)  # height /16 x width /16   
        iconv4 = self.iconv4(torch.cat((upconv4, conv4), 1))
        
        upconv3 = self.upconv3(iconv4)  # height /8 x width /8
        iconv3 = self.iconv3(torch.cat((upconv3, conv3), 1))
        disp3 = scale * self.disp3(iconv3) # height /8 x width /8
        udisp3 = F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=True) # height /4 x width /4
        
        upconv2 = self.upconv2(iconv3)  # height /4 x width /4
        iconv2 = self.iconv2(torch.cat((upconv2, conv2, udisp3), 1))
        disp2 = scale * self.disp2(iconv2)
        udisp2 = F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=True)
        
        upconv1 = self.upconv1(iconv2)  # height /2 x width /2
        iconv1 = self.iconv1(torch.cat((upconv1, conv1, udisp2), 1))
        disp1 = scale * self.disp1(iconv1)
        udisp1 = F.interpolate(disp1, scale_factor=2, mode='bilinear', align_corners=True)
        
        upconv0 = self.upconv0(iconv1)  # height x width
        iconv0 = self.iconv0(torch.cat((upconv0, udisp1), 1))
        disp0 = scale * self.disp0(iconv0)
        
        return [disp0, disp1, disp2, disp3]
        
        
    def _downConvLayer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _upConvLayer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _convLayer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _dispLayer(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def _computeCostVolume(self, reference_images, source_images, KRKiUV, Kt):
        # Input:
        #   images: samples x 3 x height x width 
        #   KRKiUV: samples x 3 x pixels
        #   Kt: samples x 3 
        # Output:
        #   cost_volume: samples x depth_samples x height x width
        
        depth_max = 50.0
        depth_min = 0.5
        steps = 256
        idepth_min = 1.0 / depth_max
        idepth_max = 1.0 / depth_min
        idepth_step = (idepth_max - idepth_min) / (steps-1)
        
        cost_volume = torch.zeros(reference_images.shape[0], steps, reference_images.shape[2], reference_images.shape[3]).to('cuda:0')
        
        for i in range(steps):
            depth = 1.0 / (idepth_min + i * idepth_step)
            warped = self._warpImage(source_images, depth, KRKiUV, Kt)
            abs_residual = torch.sum(torch.abs(reference_images - warped), dim=1)
            cost_volume[:,i,:,:] = abs_residual
        
        return cost_volume
    
    def _computeDehazingCostVolume(self, reference_images, source_images, KRKiUV, Kt, A, beta):
        # Input:
        #   images: samples x 3 x height x width 
        #   KRKiUV: samples x 3 x pixels
        #   Kt: samples x 3 
        # Output:
        #   cost_volume: samples x depth_samples x height x width
        
        depth_max = 50.0
        depth_min = 0.5
        steps = 256
        idepth_min = 1.0 / depth_max
        idepth_max = 1.0 / depth_min
        idepth_step = (idepth_max - idepth_min) / (steps-1)
        
        cost_volume = torch.zeros(reference_images.shape[0], steps, reference_images.shape[2], reference_images.shape[3]).to('cuda:0')
        
        for i in range(steps):
            depth = 1.0 / (idepth_min + i * idepth_step)
            warped_img, warped_depth = self._warping(source_images, depth, KRKiUV, Kt)
            
            reference_j = (reference_images - A.unsqueeze(2).unsqueeze(3)) / (torch.exp(-depth * beta.unsqueeze(-1).unsqueeze(-1))+1.0e-6) + A.unsqueeze(2).unsqueeze(3)
            warped_j = (warped_img - A.unsqueeze(2).unsqueeze(3)) / (torch.exp(-warped_depth * beta.unsqueeze(-1).unsqueeze(-1))+1.0e-6) + A.unsqueeze(2).unsqueeze(3)
            
            abs_residual = torch.sum(torch.abs(reference_j - warped_j), dim=1)
            
            for c in range(3):
                abs_residual[reference_j[:,c,:,:] < 0] = 3
                abs_residual[warped_j[:,c,:,:] < 0] = 3
                abs_residual[reference_j[:,c,:,:] > 1] = 3
                abs_residual[warped_j[:,c,:,:] > 1] = 3
            
            cost_volume[:,i,:,:] = abs_residual
        
        return cost_volume
    
    def _warpImage(self, source_images, depth, KRKiUV, Kt):
        batch_n = source_images.shape[0]
        width = source_images.shape[3]
        height = source_images.shape[2]
        
        normalize_base = torch.FloatTensor([width/2, height/2]).to('cuda')
        normalize_base = normalize_base.unsqueeze(0).unsqueeze(-1)
        
        warped_uv = depth * KRKiUV + Kt.unsqueeze(-1) # samples x 3 x pixels
        demon = warped_uv[:,2,:].unsqueeze(1)
        warped_uv = warped_uv[:,0:2,:] / (demon + 1e-6)
        warped_uv = (warped_uv - normalize_base) / normalize_base
        warped_uv = warped_uv.view(batch_n, 2, width, height)
        warped_uv = warped_uv.transpose(1,3)
        warped = F.grid_sample(source_images, warped_uv)
        
        
        return warped
    
    def _warping(self, source_images, depth, KRKiUV, Kt):
        batch_n = source_images.shape[0]
        width = source_images.shape[3]
        height = source_images.shape[2]
        
        normalize_base = torch.FloatTensor([width/2, height/2]).to('cuda')
        normalize_base = normalize_base.unsqueeze(0).unsqueeze(-1)
        
        warped_uv = depth * KRKiUV + Kt.unsqueeze(-1) # samples x 3 x pixels
        demon = warped_uv[:,2,:].unsqueeze(1)
        demon2 = warped_uv[:,0:2,:].clone()
        warped_uv = demon2 / (demon + 1e-6)
        warped_uv = (warped_uv - normalize_base) / normalize_base
        warped_uv = warped_uv.view(batch_n, 2, width, height)
        warped_uv = warped_uv.transpose(1,3)
        warped_img = F.grid_sample(source_images, warped_uv)
        
        depth_from_reference = demon.view(batch_n, 1, width, height)
        depth_from_reference = depth_from_reference.transpose(2, 3)
        warped_depth = F.grid_sample(depth_from_reference, warped_uv)
        depth_from_reference[warped_depth == 0] = 0
        
        return warped_img, depth_from_reference