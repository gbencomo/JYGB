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
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    network = UNet(input_filters=3,
                    filters=unet_filters,
                    N=convolutions).to(device)
    network = torch.nn.DataParallel(network)
    network.load_state_dict(torch.load(checkpoint.name))
    network.eval()
    
    seed = int(''.join(i for i in checkpoint.name if i.isdigit()))

    D = Dataset(dataset, seed, train=False, augment=False)
    
    img = torch.tensor(D[index][0]).unsqueeze(0)
    pred_map = network(img)
    
    n_objects = torch.sum(pred_map).item() / 100
    n_true = np.sum(D[index][1]) / 100
    
    print(f"The number of cells counted: {n_objects}")
    print(f'True number of cells: {n_true}')
    
    img = D.images[index] / 255.
    
    true_map = D.labels[index]
    
    if visualize:
        _visualize(img, true_map, pred_map.squeeze().cpu().detach().numpy().transpose())

def _visualize(img, true_map, pred_map):
    """Draw a density map onto the image."""
    # keep the same aspect ratio as an input image
    #fig, ax = plt.subplots(figsize=figaspect(1.0 * img.shape[1] / img.shape[0]))
    #fig.subplots_adjust(0, 0, 1, 1)
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

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    #divider = make_axes_locatable(ax[2])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(im2, cax=cax)
  
    fig.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    predict()
