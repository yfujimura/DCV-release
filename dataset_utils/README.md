## Download video captured in foggy scene
We provided a video captured in actual foggy scenes. A sparse 3D point cloud and camera parameters of each frame were estimated by [COLMAP](https://colmap.github.io/){:target=``_blank``} with sequential matching and default hyperparameters. At ```colmap```, please run:

```$ bash download_shirouma2.sh```

We also provided a useful script that extracts camera parameters and sparse depth maps from COLMAP output.

```$ python save_sparse_depth.py -o shirouma2_sparse -c colmap/shirouma2``` 

Each data can be converted to a single npy file, which is loaded by a data loader.

```$ python create_dataset.py -i colmap/shirouma2/images -c shirouma2_sparse/ -o test_data -f 37```

## Use your own data
If you want to use your own data, please apply COLMAP to estimate camera parameters and a sparse 3D point cloud as preprocessing. Then you can use above scripts to make input data of the network.