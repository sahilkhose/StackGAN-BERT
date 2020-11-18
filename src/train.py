"""Test a model and generate submission CSV.

Usage:
    > python train.py --load_path PATH --name NAME
    where
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the train run

Authors:
    Abhiraj Tiwari (abhirajtiwari@gmail.com)
    Sahil Khose (sahilkhose18@gmail.com)
"""
import args
# import config
import dataset
import engine
import layers
import util

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import torch


from model import AmazingModel

from sklearn import metrics
from sklearn import model_selection
from torch.utils.tensorboard import summary
from torch.utils.tensorboard import FileWriter
from torch.autograd import Variable

print("__"*80)
print("Imports Done...")


def check_dataset(training_set):
    t, i, b = training_set[1]
    print("Bert emb shape: ", t.shape)
    print("bbox: ", b)
    plt.imshow(i)
    plt.show()
    print("__"*80)

def load_stage1(args):
    #* Init models and weights:
    from layers import Stage1Generator, Stage1Discriminator
    netG = Stage1Generator()
    netG.apply(engine.weights_init)

    netD = Stage1Discriminator()
    netD.apply(engine.weights_init)

    #* Load saved model:
    if args.NET_G_path != "":
        netG.load_state_dict(torch.load(args.NET_G_path))
        print("Generator loaded from: ", args.NET_G_path)
    if args.NET_D_path != "":
        netD.load_state_dict(torch.load(args.NET_D_path))
        print("Discriminator loaded from: ", args.NET_D_path)

    #* Load on device:
    if args.device == "cuda":
        netG.cuda()
        netD.cuda()

    print("__"*80)
    print(netG)
    print("__"*80)
    print(netD)
    print("__"*80)

    return netG, netD


def load_stage2(args):
    #* Init models and weights:
    from layers import Stage2Generator, Stage2Discriminator, Stage1Generator
    Stage1_G = Stage1Generator()
    netG = Stage2Generator(Stage1_G)
    netG.apply(engine.weights_init)

    netD = Stage2Discriminator()
    netD.apply(engine.weights_init)

    #* Load saved model:
    if args.NET_G_path != "":
        netG.load_state_dict(torch.load(args.NET_G_path))
        print("Generator loaded from: ", args.NET_G_path)
    elif args.STAGE1_G_path != "":
        netG.Stage1_G.load_state_dict(torch.load(args.STAGE1_G_path))
        print("Generator loaded from: ", args.STAGE1_G_path)
    else:
        print("Please give the Stage 1 generator path")
    
    if args.NET_D_path != "":
        netD.load_state_dict(torch.load(args.NET_D_path))
        print("Discriminator loaded from: ", args.NET_D_path)

    #* Load on device:
    if args.device == "cuda":
        netG.cuda()
        netD.cuda()

    print("__"*80)
    print(netG)
    print("__"*80)
    print(netD)
    print("__"*80)

    return netG, netD


def run(args, stage):
    if stage == 1:
        netG, netD = load_stage1(args)
    else:
        netG, netD = load_stage2(args)

    # Setting up device
    device = torch.device(args.device)

    # Load model
    netG.to(device)
    netD.to(device)

    nz = args.n_z
    batch_size = args.train_bs
    noise = Variable(torch.FloatTensor(batch_size, nz)).to(device)
    with torch.no_grad():
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1)).to(device) # volatile=True
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1)).to(device)
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0)).to(device)

    gen_lr = args.TRAIN_GEN_LR
    disc_lr = args.TRAIN_DISC_LR

    lr_decay_step = args.TRAIN_LR_DECAY_EPOCH

    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.TRAIN_DISC_LR, betas=(0.5, 0.999))

    netG_para = []
    for p in netG.parameters():
        if p.requires_grad:
            netG_para.append(p)
    optimizerG = torch.optim.Adam(netG_para, lr=args.TRAIN_GEN_LR, betas=(0.5, 0.999))

    count = 0

    training_set = dataset.CUBDataset(pickl_file=args.train_filenames, emb_dir=args.bert_annotations_dir, img_dir=args.images_dir)
    train_data_loader = torch.utils.data.DataLoader(training_set, batch_size=args.train_bs, num_workers=args.train_workers)
    # check_dataset(training_set)
    
    testing_set = dataset.CUBDataset(pickl_file=args.test_filenames, emb_dir=args.bert_annotations_dir, img_dir=args.images_dir)
    test_data_loader = torch.utils.data.DataLoader(testing_set, batch_size=args.test_bs, num_workers=args.test_workers)
    # check_dataset(testing_set)

    # best_accuracy = 0

    # summary_writer = FileWriter(args.log_dir) ## TODO

    for epoch in range(args.TRAIN_MAX_EPOCH):
        print("__"*80)
        start_t = time.time()

        if epoch % lr_decay_step == 0 and epoch > 0:
            gen_lr *= 0.5
            for param_group in optimizerG.param_groups:
                param_group["lr"] = gen_lr
            disc_lr *= 0.5
            for param_group in optimizerD.param_groups:
                param_group["lr"] = disc_lr
        
        # engine.train_fn()
        # d_loss, g_loss = engine.train_fn(train_data_loader, netD, netG, optimizerD, optimizerG, device, epoch) # TODO
        
        end_t = time.time()
        
    #     print("Losses") # TODO
    #     if epoch % self.snapshot_interval == 0: # TODO
    #         save_model(netG, netD, epoch, self.model_dir)
        break
    
    # save_model(netG, netD, self.max_epoch, self.model_dir) # TODO
    # summary_writer.close()

 
if __name__ == "__main__":
    run(args.get_all_args(), stage=1)
