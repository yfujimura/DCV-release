import os
import cv2
import numpy as np
import argparse
from scipy.spatial.transform import Rotation
import statistics
import matplotlib.pyplot as plt
import shutil

# This code creates camera parameters and sparse 3D points from colmap sparse output. 
# Please specify a colmap projetct path as input.
# npy files corresponding each input frame will be created under an output path.

# exmaple:
# python save_sparse_depth.py -o shirouma2_sparse -c colmap/shirouma2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_path")
    parser.add_argument("-c", "--colmap_project_path")
    args = parser.parse_args()
    return args

def get_points2d(fn):
    img_list = []
    points2d_list = []
    R_list = []
    t_list = []
    
    with open(fn, "r") as f:
        lines = f.readlines()
        for i in range(1, len(lines), 2):
            points2d = {}
            line = lines[i]
            tokens = line.split(" ")
            img_idx = int(os.path.splitext(tokens[9])[0])
            img_list.append(img_idx)
            
            q = [float(tokens[2]), float(tokens[3]), float(tokens[4]), float(tokens[1])]
            rot = Rotation.from_quat(q)
            R = rot.as_dcm()
            t = np.array([float(x) for x in tokens[5:8]])
            R_list.append(R)
            t_list.append(t)
            
            line = lines[i+1]
            tokens = line.split(" ")
            tokens = [float(t) for t in tokens]
            for j in range(0, len(tokens), 3):
                if tokens[j+2] !=-1:
                    points2d[int(tokens[j+2])] = [tokens[j], tokens[j+1]]
                    
            points2d_list.append(points2d)
                    
    return img_list, points2d_list, R_list, t_list

def get_points3d(fn):
    points3d = {}
    img_id_list = {}
    pixel_id_list = {}
    
    with open(fn, "r") as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            line = lines[i]
            tokens = line.split(" ")
            idx = int(tokens[0])
            # [x, y, z, error]
            point = [float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[7])]
            points3d[idx] = point
            img_pixel = tokens[8:]
            img_pixel = [int(x) for x in img_pixel]
            img_id = [img_pixel[j] for j in range(0, len(img_pixel), 2)]
            pixel_id = [img_pixel[j] for j in range(1, len(img_pixel), 2)]
            img_id_list[idx] = img_id
            pixel_id_list[idx] = pixel_id
            
    return points3d, img_id_list, pixel_id_list

def get_intrinsic(fn):
    K = np.zeros((3,3))
    K[2,2] = 1
    with open(fn, "r") as f:
        lines = f.read().splitlines()
        line = lines[1]
        tokens = line.split(" ")
        width = int(tokens[2])
        height = int(tokens[3])
        focal_length = float(tokens[4])
        cx = float(tokens[5])
        cy = float(tokens[6])
        K[0,0] = focal_length
        K[1,1] = focal_length
        K[0,2] = cx
        K[1,2] = cy
    return width, height, K
        
        


def main():
    args = parse_args()
    
    out_root_path = args.out_path
    colmap_project_path = args.colmap_project_path
    
    if not os.path.exists(out_root_path):
        os.mkdir(out_root_path)
    
    width, height, K = get_intrinsic(os.path.join(colmap_project_path, "cameras.txt"))
    img_list, points2d_list, R_list, t_list = get_points2d(os.path.join(colmap_project_path, "images.txt"))
    points3d, img_id_list, pixel_id_list = get_points3d(os.path.join(colmap_project_path, "points3D.txt"))
    
    np.save(os.path.join(out_root_path, "K"), K)
    
    for img_id in range(0, len(img_list), 1):
        points2d = points2d_list[img_id]
        R = R_list[img_id]
        t = t_list[img_id]
        
        feature_points = np.zeros((height,width,3))
        for j in points3d:
            if j in points2d:
                x,y = points2d[j]
                x = round(x)
                y = round(y)
                if x >= width or y >= height:
                    continue
                point = np.array(points3d[j][0:3])
                point = np.dot(R, point) + t
                feature_points[y,x] = point
                
        out_path = os.path.join(out_root_path, "{:06}".format(img_list[img_id]))
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        
        np.save(os.path.join(out_path, "sparse_points"), feature_points)
        np.save(os.path.join(out_path, "R"), R)
        np.save(os.path.join(out_path, "t"), t)
    

if __name__ == "__main__":
    main()