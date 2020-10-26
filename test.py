import os
import glob
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from dehazing_mvs_net import *
from airlight_estimator import *
from parameters_estimation import *

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dcv_mvsnet_checkpoint")
    parser.add_argument("--airlight_checkpoint")
    parser.add_argument("--beta_min", type=float, default=0.01)
    parser.add_argument("--beta_max", type=float, default=0.1)
    parser.add_argument("--beta_steps", type=int, default=10)
    parser.add_argument("--delta_A", type=float, default=0.05)
    parser.add_argument("--delta_beta", type=float, default=0.01)
    parser.add_argument("--A_steps", type=int, default=4)
    parser.add_argument("--beta_steps2", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    
    args = parser.parse_args()
    return args

class MVSDataset(Dataset):
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_list = glob.glob(os.path.join(root_dir, "*.npy"))
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = np.load(os.path.join(self.root_dir, "{:06}.npy".format(idx)))
        output = [data[i] for i in range(data.shape[0])]
        output.append(idx)
        
        return output
    
    
def main():
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    height = 192
    width = 256
    pixel_coordinate = np.indices([width, height]).astype(np.float32)
    pixel_coordinate = np.concatenate((pixel_coordinate, np.ones([1, width, height]).astype(np.float32)), axis=0)
    pixel_coordinate = np.reshape(pixel_coordinate, [3, -1])
    pixel_coordinate = torch.from_numpy(pixel_coordinate)
    
    # load models
    dehazing_mvs_net = DehazingMVSNet()
    dehazing_mvs_net.to('cuda')
    params = torch.load(args.dcv_mvsnet_checkpoint)
    dehazing_mvs_net.load_state_dict(params)
    
    airlight_estimator = AirlightEstimator()
    airlight_estimator.to('cuda')
    params = torch.load(args.airlight_checkpoint)
    airlight_estimator.load_state_dict(params)
    
    # load test data
    test_dataset = MVSDataset("dataset_utils/test_data")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    
    
    # depth and parameters estimation
    dehazing_mvs_net.eval()
    airlight_estimator.eval()
    beta_range = (args.beta_min, args.beta_max)
    beta_steps = args.beta_steps
    for reference_images, source_images, KRKi, Kt, feature_points, idx in test_loader:
        reference_images = reference_images.transpose(1,2).transpose(1,3).to('cuda')  # samples x 3 x heigh x width
        source_images = source_images.transpose(1,2).transpose(1,3).to('cuda')  # samples x 3 x heigh x width
        KRKi = KRKi.to('cuda') # samples x 3 x 3
        Kt = Kt.to('cuda') # samples x 3 
        feature_points = feature_points.transpose(1,2).transpose(1,3).to('cuda')  # samples x 4 x heigh x width
        index = int(idx[0].numpy())
        
        with torch.no_grad():
            A_pred = airlight_estimator(reference_images).item()
            
        print("A0 =", A_pred)
            
        feature_depth = feature_points[:, 2, :, :].unsqueeze(1) 
        masks = feature_depth.detach().clone()
        masks[feature_depth > 0] = 1
        masks[feature_depth <= 0] = 0
        
        pixel_coordinate_batch = pixel_coordinate.expand(reference_images.shape[0], pixel_coordinate.shape[0], pixel_coordinate.shape[1])
        pixel_coordinate_batch = pixel_coordinate_batch.to('cuda')
        KRKiUV = torch.bmm(KRKi, pixel_coordinate_batch)    # samples x 3 x pixels
        
        print("search beta in", beta_range)
        errors, beta_min_idx, beta, predict, predicts = search_beta(dehazing_mvs_net,
                                                                    reference_images, source_images, KRKiUV, Kt, A_pred,
                                                                    beta_range, beta_steps,
                                                                    feature_depth, masks)
        print("beta0 = ", beta)
        
        A_range = (A_pred-args.delta_A, A_pred+args.delta_A)
        A_steps = args.A_steps
        beta_range2 = (beta-args.delta_beta, beta+args.delta_beta)
        beta_steps2 = args.beta_steps2
        print("search A and beta in", A_range, beta_range2)
        error_map, A_min_idx, beta_min_idx2, A, beta2, predict, predicts = search_A_beta(dehazing_mvs_net,
                                                                                         reference_images, source_images, KRKiUV, Kt,
                                                                                         A_range, A_steps,
                                                                                         beta_range2, beta_steps2,
                                                                                         feature_depth, masks)
        print("A = ", A)
        print("beta =", beta2)
        
    # visualize result
    r_img = reference_images.transpose(1,2).transpose(2,3).to('cpu').detach().numpy()
    s_img = source_images.transpose(1,2).transpose(2,3).to('cpu').detach().numpy()
    p_disp = predict[0][:,0,:,:].to('cpu').detach().numpy()
    sparse_depth = feature_depth[:,0,:,:].to('cpu').detach().numpy()
    
    p_disp_log = np.log(p_disp[0]+1e-8)
    p_min = np.min(p_disp_log)*0.6
    p_max = np.max(p_disp_log) 
    
    depth_point = sparse_depth[0]
    max_depth = np.max(depth_point)
    depth_point = (depth_point * 255 / max_depth).astype(np.uint8)
    depth_point2 = depth_point.copy()
    for y in range(height):
        for x in range(width):
            if depth_point[y,x] != 0:
                cv2.circle(depth_point2, (x,y), 2, int(depth_point[y,x]), -1)
    depth_point2 = depth_point2.astype(np.float32) * max_depth / 255
    depth_point2[depth_point2==0] = np.nan
    disp_point2_log = np.log(1/(depth_point2+1e-8))
    
    plt.subplot(2,2,1)
    plt.title("Reference image")
    plt.imshow(r_img[0])
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    
    plt.subplot(2,2,2)
    plt.title("Source image")
    plt.imshow(s_img[0])
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    
    plt.subplot(2,2,3)
    plt.title("Output depth")
    plt.imshow(p_disp_log, cmap='jet', clim=(p_min,p_max))
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    
    plt.subplot(2,2,4)
    plt.title("SfM depth")
    plt.imshow(disp_point2_log, cmap='jet', clim=(p_min,p_max))
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    
    plt.savefig("result.png")
    
    print("save => result.png")


if __name__ == "__main__":
    main()