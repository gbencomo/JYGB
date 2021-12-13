import click

import torch
import numpy as np
import matplotlib.pyplot as plt

from data import Dataset
from model import UNet

@click.command()
@click.option('-d', '--dataset',
        type=click.Choice(['vgg', 'mbm', 'dcc', 'adi']),
        required=True,
        help='Dataset to pull image from (HDF5).')
@click.option('-i', '--index', default=np.random.randint(10),
        help='Image index to visualize.')
@click.option('-c', '--checkpoint',
              type=click.File('r'),
              required=True,
              help='A path to a checkpoint with weights.')
@click.option('-u', '--unet_filters', default=64,
              help='Number of filters for U-Net convolutional layers.')
@click.option('-co', '--convolutions', default=2,
              help='Number of layers in a convolutional block.')
@click.option('-v', '--visualize',
              is_flag=True,
              help="Visualize predicted density map.")

def predict(    dataset: str,
                index: int,
                checkpoint: str,
                unet_filters: int,
                convolutions: int,
                visualize: bool):

    # run on gpu if available    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # initialize network as it was turning training run
    network = UNet(input_filters=3,
                    filters=unet_filters,
                    N=convolutions).to(device)
    network = torch.nn.DataParallel(network)
    network.load_state_dict(torch.load(checkpoint.name))
    network.eval()
    
    # the seed determines the test-train split.  pull the seed from the file name
    seed = int(''.join(i for i in checkpoint.name if i.isdigit()))

    # pull test data using the seed
    D = Dataset(dataset, seed, train=False, augment=False)
    
    # get image from test, predict
    img = torch.tensor(D[index][0]).unsqueeze(0)
    pred_map = network(img)
    
    # calculate cell counts for pred and true
    n_objects = torch.sum(pred_map).item() / 100
    n_true = np.sum(D[index][1]) / 100
    
    print(f"The number of cells counted: {n_objects}")
    print(f'True number of cells: {n_true}')
    
    # pull raw image
    img = D.images[index] / 255.
    
    # pull ground truth label
    true_map = D.labels[index]
    
    if visualize:
        _visualize(img, true_map, pred_map.squeeze().cpu().detach().numpy().transpose())

def _visualize(img, true_map, pred_map):
    """Plot the raw image, the predicted density map, and the ground truth density map."""
    
    fig, ax = plt.subplots(1, 3)

    im0 = ax[0].imshow(img)
    ax[0].set_title("Input Image", fontsize=8)
    ax[0].axis('off')
    
    im1 = ax[1].imshow(pred_map)
    ax[1].set_title("Predicted Density Map", fontsize=8)
    ax[1].axis('off')

    im2 = ax[2].imshow(true_map)
    ax[2].set_title("Groundtruth Density Map", fontsize=8)
    ax[2].axis('off')

    fig.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    predict()
