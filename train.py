import argparse
import data

from config import config as cfg
from config import (cfg_from_list, assert_and_infer_cfg, print_cfg)


def train():
    dataset = data.Dataset(cfg.DATASET, cfg.RNG_SEED)
    imgs, labels = dataset.preprocessing(
        training=True, augment=True, batch_size=cfg.TRAIN.BATCH_SIZE, num_epochs=cfg.TRAIN.EPOCH)    
    print('Training set pre-processing complete.')
    print(f'Training features shape: {imgs.shape}')
    print(f'Training labels shape: {labels.shape}')
def main():
    
    # initialize
    parser = argparse.ArgumentParser(description='Cell-Counting Model Training')
    
    # TODO: add arguments for config files...

    parser.add_argument('opts', help='see config.py for all options',
                        default=None, nargs=argparse.REMAINDER)

    # parse command-line arguments
    args = parser.parse_args()

    # process options
    if args.opts is not None:
        cfg_from_list(args.opts)    

    # check dataset and then print configuration        
    assert_and_infer_cfg()
    print_cfg()

    # TODO: train
    train()
    # TODO: bn_update

    # TODO: test
        
if __name__ == '__main__':
    main()
