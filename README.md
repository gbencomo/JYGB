# Cell Detection for 2-D Microscopy Images with U-Net based CNNs

This repository contains the computer vision (COS429) final project of Gianluca Bencomo and John Yang.  This project was conducted at Princeton University as a part of our graduate studies in computer science. The paper submitted, code, and data used are all included in this repository.

## Dependencies
- Python 3.8
- Pytorch 1.9
- Numpy 1.20
- Matplotlib 3.3
- Click 8.0
- Time 3.10
- H5py 3.5

## Data
All the four datasets are included in this repository for convenience.

The dot annotations were processed using `scipy.ndimage.gaussian_filter`.

Original Datasets:
- [VGG](http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip)
- [MBM & ADI](https://github.com/ieee8023/countception)
- [DCC](https://github.com/markmarsden/DublinCellDataset)

## Run

From the root folder, run
```
bash run.sh [2D_dataset]
```
For example, the following code will run on `vgg` dataset. 
```
bash run.sh vgg
```
Each time the training and test set will be randomly split by a random seed appended in the output folder. The model and training logs can both be located in the output folder.
