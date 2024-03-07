'''
    File name: train.py
    Author: Gabriel Moreira, Samidhya Sarker
    Date last modified: 03/01/2024
    Python Version: 3.12
'''

import os
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as ttf

from resnet import ResNet18, ResNet50
from utils import getNumTrainableParams
from trainer import Trainer
from torch.utils.data import Dataset, DataLoader, random_split

if __name__ == '__main__':

    TRAIN_DIR  = "~/dev/dat/personai_icartoonface_rectrain/icartoonface_rectrain_split/train"
    #TRAIN_DIR  = "./classification/train_subset/train_subset/train"
    DEV_DIR    = "~/dev/dat/personai_icartoonface_rectrain/icartoonface_rectrain_split/val"
    
    # Hyperparams
    RESUME     = True
    NAME       = 'rn50_v1'
    EPOCHS     = 100
    BATCH_SIZE = 64
    LR         = 0.1

    config = {'name'       : NAME,
              'epochs'     : EPOCHS,
              'batch_size' : BATCH_SIZE,
              'lr'         : LR,
              'resume'     : RESUME}

    print('Experiment ' + config['name'])

    # If experiment folder doesn't exist create it
    if not os.path.isdir(config['name']):
        os.makedirs(config['name'])
        print("Created experiment folder : ", config['name'])
    else:
        print(config['name'], "folder already exists.")

    if torch.cuda.is_available():
        device =  torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device == "cuda":
        torch.cuda.empty_cache()

    train_transforms = [ttf.ToTensor(),
                        ttf.Normalize(mean=[0.5677, 0.5188, 0.4885], std=[0.2040, 0.2908, 0.2848]),
                        ttf.RandomHorizontalFlip(p=0.5),
                        ttf.RandomAdjustSharpness(sharpness_factor=2),
                        ttf.ColorJitter(brightness=.3, hue=.1, saturation=0.2),
                        ttf.RandomRotation(degrees=(-30, 30)),
                        ttf.RandomPerspective(distortion_scale=0.4, p=0.75)]

    dev_transforms = [ttf.ToTensor()]

    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=ttf.Compose(train_transforms))
    dev_dataset   = torchvision.datasets.ImageFolder(DEV_DIR, transform=ttf.Compose(dev_transforms))

    # smallset = torch.utils.data.Subset(train_dataset, list(range(1000)))
    # train_size = int(0.8 * len(smallset))
    # val_size = len(smallset) - train_size
    # smaller_train_dataset, smaller_dev_dataset = random_split(smallset, [train_size, val_size])

    train_loader  = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    dev_loader    = DataLoader(dev_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)

    # Choose the model
    model = ResNet50().to(device)

    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * config['epochs']))
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model,
                      config['epochs'],
                      optimizer,
                      scheduler,
                      criterion,
                      train_loader, 
                      dev_loader,
                      device,
                      config['name'],
                      config['resume'])

    # Verbose
    print('Running on', device)
    print('Train - {} batches of {} images'.format(len(train_loader), config['batch_size']))
    print('  Val - {} batches of {} images'.format(len(dev_loader), config['batch_size']))
    print('Number of trainable parameters: {}'.format(getNumTrainableParams(model)))
    print(model)

    trainer.fit() 

