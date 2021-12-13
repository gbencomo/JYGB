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
bash run.sh [dataset]
```
For example, the following code will run on `vgg` dataset. 
```
bash run.sh vgg
```
Each time the training and test set will be randomly split by a random seed appended in the output folder. The model and training logs can both be located in the output folder. Training configurations can be changed in `run.sh` or by passing different arguments when runing `python train.py` directly.
```
Usage: train.py [OPTIONS]

Options:
  -d, --dataset [vgg|mbm|dcc|adi]
                                  Dataset to train model on (HDF5).
                                  [required]

  -lr, --learning_rate FLOAT      Initial learning rate.
  -e, --epochs INTEGER            Number of training epochs.
  -b, --batch_size INTEGER        Batch size for both training and validation.
  -a, --augment                   Augment training data.
  -uf, --unet_filters INTEGER     Number of filters for U-Net convolutional
                                  layers.

  -c, --convolutions INTEGER      Number of layers in a convolutional block.
  -p, --plot                      Generate a live plot.
  -wd, --weight_decay FLOAT       Weight decay.
  -m, --momentum FLOAT            Momentum.
  -o, --optim TEXT                Optimizer for training (Options: AdamW,
                                  RMSprop, SDG).

  -s, --seed INTEGER              Seed for train/test split.
  -sc, --scheduler TEXT           Learning rate scheduler.
  -l, --loss_function TEXT        Loss function to use.
  -sp, --save_path TEXT           Specify the save path to which
                                  models/results should be saved.  [required]

  --help                          Show this message and exit.
```

## Predictions

The `predict.py` script is provided to run a trained model on any given input image.

```
Usage: predict.py [OPTIONS]

Options:
  -d, --dataset [vgg|mbm|dcc|adi]
                                  Dataset to pull image from (HDF5).
                                  [required]

  -i, --index INTEGER             Image index to visualize.
  -c, --checkpoint FILENAME       A path to a checkpoint with weights.
                                  [required]

  -u, --unet_filters INTEGER      Number of filters for U-Net convolutional
                                  layers.

  -co, --convolutions INTEGER     Number of layers in a convolutional block.
  -v, --visualize                 Visualize predicted density map.
  --help                          Show this message and exit.

```

### Examples
