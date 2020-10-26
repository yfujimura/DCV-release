import numpy as np
import os
import argparse
import cv2
import shutil
import tqdm
import statistics

# This code creates a npy file that will be loaded by a data loader.
# A created npy file contains:
#   hazy_r: reference frame, which can be specified by the argument "frame".
#   hazy_s: source image
#   Ks_Rrs_Kri: 3 x 3 array
#   Ks_trs: 3d array
#   sparse_points: h x w x 3 array. Each pixel srores a 3D point obtained by SfM. 

# exmaple:
# python create_dataset.py -i colmap/shirouma2/images -c shirouma2_sparse/ -o test_data -f 37

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path")
    parser.add_argument("-c", "--colmap_output_path")
    parser.add_argument("-o", "--output_path")
    parser.add_argument("-f", "--frame", type=int)
    args = parser.parse_args()
    return args

def resize_sparse_points(sparse_points, out_size):
    h, w, c = sparse_points.shape
    scale_x = out_size[0] / w
    scale_y = out_size[1] / h
    
    resized_points = np.zeros((out_size[1], out_size[0], c))
    for y in range(h):
        y_scaled = y *scale_y
        y_scaled = round(y_scaled)
        for x in range(w):
            x_scaled = x * scale_x
            x_scaled = round(x_scaled)
            
            if 0 <= x_scaled < out_size[0] and 0 <= y_scaled < out_size[1]:
                resized_points[y_scaled, x_scaled] = sparse_points[y,x]
                
    return resized_points
    
    

def create_dataset(image_path, colmap_output_path, output_path, image_size, frame=0, interval=5, scale=1.0):
    
    i_sample = 0
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    frames = os.listdir(image_path)
    frames = len(frames)
    
    K = np.load(os.path.join(colmap_output_path, "K.npy"))
    K[0,:] = K[0,:] * image_size[0] / 1920.
    K[1,:] = K[1,:] * image_size[1] / 1080.
    
    vr = frame
    vs = vr + interval
    if vs >= frames:
        vs = vr - interval
        
    hazy_r = cv2.cvtColor(cv2.imread(os.path.join(image_path, "{:06}.png".format(vr))), cv2.COLOR_BGR2RGB)
    hazy_s = cv2.cvtColor(cv2.imread(os.path.join(image_path, "{:06}.png".format(vs))), cv2.COLOR_BGR2RGB)
    hazy_r = cv2.resize(hazy_r, image_size, interpolation=cv2.INTER_AREA)
    hazy_s = cv2.resize(hazy_s, image_size, interpolation=cv2.INTER_AREA)
    hazy_r = hazy_r.astype(np.float32) / 255.
    hazy_s = hazy_s.astype(np.float32) / 255.
    
    Rr = np.load(os.path.join(colmap_output_path, "{:06}".format(vr), "R.npy"))
    tr = np.load(os.path.join(colmap_output_path, "{:06}".format(vr), "t.npy"))
    Rs = np.load(os.path.join(colmap_output_path, "{:06}".format(vs), "R.npy"))
    ts = np.load(os.path.join(colmap_output_path, "{:06}".format(vs), "t.npy"))
    
    sparse_points = np.load(os.path.join(colmap_output_path, "{:06}".format(vr), "sparse_points.npy"))
    sparse_points = resize_sparse_points(sparse_points, image_size)
    
    sparse_points = sparse_points * scale
    tr = tr * scale
    ts = ts * scale
   
    RrT = Rr.T
    RsT = Rs.T
    Rrs = np.dot(Rs, RrT)
    RrT_tr = np.dot(RrT, tr)
    trs = -np.dot(Rs, RrT_tr) + ts
    Rrs_Kri = np.dot(Rrs, np.linalg.inv(K))
    Ks_Rrs_Kri = np.dot(K, Rrs_Kri).astype(np.float32)
    Ks_trs = np.dot(K, trs).astype(np.float32)
        
    sparse_points = sparse_points.astype(np.float32)
        
    sample = [hazy_r, hazy_s, Ks_Rrs_Kri, Ks_trs, sparse_points]
    sample = np.array(sample)
    np.save(os.path.join(output_path, '{:06}'.format(i_sample)), sample)
    
                
def main():
    args = parse_args()
    image_path = args.image_path
    colmap_output_path = args.colmap_output_path
    output_path = args.output_path
    frame = args.frame
    
    create_dataset(image_path, colmap_output_path, output_path, (256, 192), frame=frame, interval=5, scale=1)
        

if __name__ == "__main__":
    main()   