import numpy as np
from tqdm import tqdm

import torch

def huber_loss(x, y, d):
    loss_map = torch.abs(x - y)
    tmp = loss_map.detach().clone()
    
    loss_map[tmp <= d] = tmp[tmp <= d]**2 / 2
    loss_map[tmp > d] = d * (tmp[tmp > d] - d / 2)
    
    loss = torch.sum(loss_map)
    
    return loss_map, loss

def loss_for_search(depth, feature_depth, masks, d=5, loss_fn="L1", a=0.5):
    n, c, h, w = depth.shape
    depth2 = torch.zeros(n, c, h+2*d, w+2*d).to('cuda')
    depth2[:,:,d:d+h,d:d+w] = depth
    
    if loss_fn == "L1":
        loss_map_0 = torch.abs(masks*(depth2[:,:,d:d+h,d:d+w]) - masks*feature_depth)
        loss_map_xl = torch.abs(masks*(depth2[:,:,d:d+h,0:w]) - masks*feature_depth)
        loss_map_xr = torch.abs(masks*(depth2[:,:,d:d+h,2*d:2*d+w]) - masks*feature_depth)
        loss_map_yu = torch.abs(masks*(depth2[:,:,0:h,d:d+w]) - masks*feature_depth)
        loss_map_yb= torch.abs(masks*(depth2[:,:,2*d:2*d+h,d:d+w]) - masks*feature_depth)
    if loss_fn == "Huber":
        loss_map_0, _ = huber_loss(masks*(depth2[:,:,d:d+h,d:d+w]), masks*feature_depth, a)
        loss_map_xl, _ = huber_loss(masks*(depth2[:,:,d:d+h,0:w]), masks*feature_depth, a)
        loss_map_xr, _ = huber_loss(masks*(depth2[:,:,d:d+h,2*d:2*d+w]), masks*feature_depth, a)
        loss_map_yu, _ = huber_loss(masks*(depth2[:,:,0:h,d:d+w]), masks*feature_depth, a)
        loss_map_yb, _= huber_loss(masks*(depth2[:,:,2*d:2*d+h,d:d+w]), masks*feature_depth, a)
                
    loss_map = torch.cat((loss_map_0, loss_map_xl, loss_map_xr, loss_map_yu, loss_map_yb), 1)
    loss_map, _ = torch.min(loss_map, 1, keepdim=True)
    loss = torch.sum(loss_map) / torch.nonzero(masks).size(0)
    return loss.item()

def search_beta(dehazing_mvs_net,
                reference_images, source_images, KRKiUV, Kt, A,
                beta_range, beta_steps,
                feature_depth, masks,
                d=5, loss_fn="L1", a=0.5):
    
    beta_step = (beta_range[1] - beta_range[0]) / (beta_steps-1)
    
    ones = torch.ones(1,1).to('cuda')
    A = ones * A
    predicts = []
    errors = []
    for ibeta in tqdm(range(beta_steps)):
        beta_val = beta_range[0] + ibeta * beta_step
        beta = ones * beta_val
        with torch.no_grad():
            predict = dehazing_mvs_net(reference_images, 
                                       source_images, 
                                       KRKiUV, 
                                       Kt, 
                                       A, beta)
            depth = 1/(predict[0]+1.0e-8)
            predicts.append(predict)
            
            loss = loss_for_search(depth, feature_depth, masks, d=d, loss_fn=loss_fn, a=a)
            errors.append(loss)
            
    beta_min_idx = np.argmin(np.array(errors))
    beta = beta_range[0] + beta_min_idx * beta_step
    predict = predicts[beta_min_idx]
    return errors, beta_min_idx, beta, predict, predicts

def search_A_beta(dehazing_mvs_net,
                  reference_images, source_images, KRKiUV, Kt,
                  A_range, A_steps,
                  beta_range, beta_steps,
                  feature_depth, masks,
                  d=5, loss_fn="L1", a=0.5):
    
    A_step = (A_range[1] - A_range[0]) / (A_steps-1)
    beta_step = (beta_range[1] - beta_range[0]) / (beta_steps-1)
    
    ones = torch.ones(1,1).to('cuda')
    predicts = []
    error_map = np.zeros((A_steps, beta_steps))
    for iA in tqdm(range(A_steps)):
        A_val = A_range[0] + iA * A_step
        A = ones * A_val
        for ibeta in range(beta_steps):
            beta_val = beta_range[0] + ibeta * beta_step
            beta = ones * beta_val
            with torch.no_grad():
                predict = dehazing_mvs_net(reference_images, 
                                           source_images, 
                                           KRKiUV, 
                                           Kt, 
                                           A, beta)
                depth = 1/(predict[0]+1.0e-8)
                predicts.append(predict)
            
                loss = loss_for_search(depth, feature_depth, masks, d=d, loss_fn=loss_fn, a=a)
                error_map[iA, ibeta] = loss
            
    A_min_idx, beta_min_idx = np.where(error_map == np.min(error_map))
    A = A_range[0] + A_step * A_min_idx
    beta = beta_range[0] + beta_step * beta_min_idx
    idx = int(beta_min_idx + beta_steps * A_min_idx)
    predict = predicts[idx]
    
    return error_map, A_min_idx, beta_min_idx, A, beta, predict, predicts