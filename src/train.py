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
import torch


from model import AmazingModel

from sklearn import metrics
from sklearn import model_selection

print("__"*80)
print("Imports Done...")


def check_dataset(training_set):
    t, i, b = training_set[1]
    print("Bert emb shape: ", t.shape)
    print("bbox: ", b)
    plt.imshow(i)
    plt.show()


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

    training_set = dataset.CUBDataset(pickl_file=args.train_filenames, emb_dir=args.bert_annotations_dir, img_dir=args.images_dir)
    train_data_loader = torch.utils.data.DataLoader(training_set, batch_size=2, num_workers=1)
    # check_dataset(training_set)
    # print("__"*80)
    testing_set = dataset.CUBDataset(pickl_file=args.test_filenames, emb_dir=args.bert_annotations_dir, img_dir=args.images_dir)
    test_data_loader = torch.utils.data.DataLoader(testing_set, batch_size=2, num_workers=1)
    # check_dataset(testing_set)

    # Setting up device
    device = torch.device(args.device)

    # Load model
    # load_file = config.MODEL_PATH + "7_model_15.bin"
    generator1 = layers.Stage1Generator()
    generator2 = layers.Stage2Generator()

    discriminator1 = layers.Stage1Discriminator()
    discriminator2 = layers.Stage2Discriminator()

    # if os.path.exists(load_file):
    #     model.load_state_dict(torch.load(load_file))
    generator1.to(device)
    generator2.to(device)
    discriminator1.to(device)
    discriminator2.to(device)

    # Setting up training
    # num_train_steps = int(len(id_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    lr = 0.0002  # TODO decay this every 100 epochs by 0.5
    optimG1 = torch.optim.Adam(generator1.parameters(), lr, betas=(0.5, 0.999))  # Beta value from github impl
    optimD1 = torch.optim.Adam(discriminator1.parameters(), lr, betas=(0.5, 0.999))

    best_accuracy = 0

    # Main training loop
    for epoch in range(1, 10):  # config.EPOCHS+1):
        # Running train, valid, test loop every epoch
        print("__"*80)
        d_loss, g_loss = engine.train_fn(
            train_data_loader, discriminator1, generator1, optimD1, optimG1, device, epoch)
        print("losses: ", d_loss, g_loss)
        break

 
if __name__ == "__main__":
    run(args.get_all_args(), stage=1)
