import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from options import parse_args
from dataloader import create_dataloader
import os
import sys

sys.path.append(os.path.abspath('..'))
from utils import log
from agent import Agent
from models.prover import Prover

def main():
    # parse the options
    opts = parse_args()

    # create the dataloaders
    dataloader = {'train': create_dataloader('train', opts),
                  'valid': create_dataloader('valid', opts)}

    # create the model
    model = Prover(opts)
    model.to(opts.device)

    # crete the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.learning_rate)

    # load the checkpoint
    start_epoch = 0

    if opts.resume != None:
        log('loading model checkpoint from %s..' % opts.resume)
        if opts.device.type == 'cpu':
            checkpoint = torch.load(opts.resume, map_location='cpu')
        else:
            checkpoint = torch.load(opts.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['n_epoch'] + 1
        model.to(opts.device)

    agent = Agent(model, optimizer, dataloader, opts)

    for n_epoch in range(start_epoch, start_epoch + opts.num_epochs):
        log('EPOCH #%d' % n_epoch)

        # training
        loss_train = agent.train(n_epoch)

        # validation
        # if not opts.no_validation:
        #     loss_valid = agent.valid(n_epoch)

        # agent.inference(n_epoch)

if __name__ == '__main__':
    main()