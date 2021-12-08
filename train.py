import click
import torch
from torch.utils.data import DataLoader
import numpy as np
from data import Dataset
from model import UNet
from matplotlib import pyplot
from looper import Looper
import time

@click.command()
@click.option('-d', '--dataset',
        type=click.Choice(['vgg', 'mbm', 'dcc', 'adi']),
        required=True,
        help='Dataset to train model on (HDF5).')
@click.option('-lr', '--learning_rate', default=1e-3,
        help='Initial learning rate.')
@click.option('-e', '--epochs', default=150, 
        help='Number of training epochs.')
@click.option('-b', '--batch_size', default=16,
        help='Batch size for both training and validation.')
@click.option('-a', '--augment', is_flag=True, help="Augment training data.")
@click.option('-uf', '--unet_filters', default=64,
        help='Number of filters for U-Net convolutional layers.')
@click.option('-c', '--convolutions', default=2,
        help='Number of layers in a convolutional block.')
@click.option('-p', '--plot', is_flag=True, 
        help="Generate a live plot.")
@click.option('-wd', '--weight_decay', default=1e-3,
        help='Weight decay.')
@click.option('-m', '--momentum', default=0.9,
        help='Momentum.')
@click.option('-s', '--seed', default=np.random.randint(10000),
        help='Seed for train/test split.')
@click.option('-v', '--val_percent', default=0.2,
        help='Percent allocation for the validation/test set.')
@click.option('-sp', '--save_path', 
        required=True,
        help='Specify the save path to which models/results should be saved.')

def train(  dataset: str,
            learning_rate: float,
            weight_decay: float,
            momentum: float,
            epochs: int,
            batch_size: int,
            augment: bool,
            unet_filters: int,
            convolutions: int,
            plot: bool,
            seed: int,
            val_percent: float,
            save_path: str):
   
    START = time.time()
 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    
    D = {}     # training and validation HDF5-based datasets
    dataloader = {}  # training and validation dataloaders
    
    # Load train/test data 
    for mode in ['train', 'valid']:
        train = (mode == 'train')
        D[mode] = Dataset(dataset, seed, train, augment, val_percent)
        dataloader[mode] = DataLoader(D[mode], batch_size=batch_size, shuffle=True) # !!! add workers, shuffle, pin_memory???
        
    # initialize network
    network = UNet(input_filters=3, filters=unet_filters, 
                    N=convolutions).to(device)
    network = torch.nn.DataParallel(network)

    # initialize loss, optimized and learning rate scheduler
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(network.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(network.parameters(),
    #                lr=learning_rate,
    #                momentum=momentum,
    #                weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                    step_size=20,
                    gamma=0.1)

    # if plot flag is on, create a live plot (to be updated by Looper)
    if plot:
        pyplot.ion()
        fig, plots = pyplot.subplots(nrows=2, ncols=2)
    else:
        plots = [None] * 2

    # create training and validation Loopers to handle a single epoch
    train_looper = Looper(network, device, loss, optimizer,
            dataloader['train'], len(D['train']), plots[0])
    valid_looper = Looper(network, device, loss, optimizer,
            dataloader['valid'], len(D['valid']), plots[1], validation=True)
        
    current_best = np.infty
    
    print(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Weight decay:    {weight_decay}
        Training size:   {len(D['train'])}
        Validation size: {len(D['valid'])}
        Device:          {device.type}
        Random seed:     {seed}
        Save path:       {save_path}
        ''')

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n")
        
        # run training epoch and update learning rate
        train_looper.run()
        lr_scheduler.step()
        
        # run validation epoch
        with torch.no_grad():
            result = valid_looper.run()
        
        # update checkpoint if new best is reached
        if result < current_best:
            current_best = result
            torch.save(network.state_dict(), f'{save_path}/{dataset}_{seed}.pth')

            print(f"\nNew best result: {result}")

        print("\n", "-"*80, "\n", sep='')
        print(f'Current learning rate: {lr_scheduler.get_last_lr()}, Training time: {(time.time() - START):.2f} s')
        print("\n", "-"*80, "\n", sep='')

    print(f"[Training done] Best result: {current_best}")
    
if __name__ == '__main__':
    train()    
