"""Test a model and generate submission CSV.

> python3 train.py --conf ../cfg/s1.yml 

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
import torchfile


from PIL import Image
from sklearn import metrics
from sklearn import model_selection
from torch.utils.tensorboard import summary
from torch.utils.tensorboard import FileWriter
from torch.autograd import Variable

print("__"*80)
print("Imports Done...")


def load_stage1(args):
    #* Init models and weights:
    from layers import Stage1Generator, Stage1Discriminator
    if args.embedding_type == "bert":
        netG = Stage1Generator(emb_dim=768)
        netD = Stage1Discriminator(emb_dim=768)
    else:
        netG = Stage1Generator(emb_dim=1024)
        netD = Stage1Discriminator(emb_dim=1024)

    netG.apply(engine.weights_init)
    netD.apply(engine.weights_init)

    #* Load saved model:
    if args.NET_G_path != "":
        netG.load_state_dict(torch.load(args.NET_G_path))
        print("__"*80)
        print("Generator loaded from: ", args.NET_G_path)
        print("__"*80)
    if args.NET_D_path != "":
        netD.load_state_dict(torch.load(args.NET_D_path))
        print("__"*80)
        print("Discriminator loaded from: ", args.NET_D_path)
        print("__"*80)

    #* Load on device:
    if args.device == "cuda":
        netG.cuda()
        netD.cuda()

    print("__"*80)
    print("GENERATOR:")
    print(netG)
    print("__"*80)
    print("DISCRIMINATOR:")
    print(netD)
    print("__"*80)

    return netG, netD


def load_stage2(args):
    #* Init models and weights:
    from layers import Stage2Generator, Stage2Discriminator, Stage1Generator
    if args.embedding_type == "bert":
        Stage1_G = Stage1Generator(emb_dim=768)
        netG = Stage2Generator(Stage1_G, emb_dim=768)
        netD = Stage2Discriminator(emb_dim=768)
    else:
        Stage1_G = Stage1Generator(emb_dim=1024)
        netG = Stage2Generator(Stage1_G, emb_dim=1024)
        netD = Stage2Discriminator(emb_dim=1024)
    netG.apply(engine.weights_init)
    netD.apply(engine.weights_init)

    #* Load saved model:
    if args.NET_G_path != "":
        netG.load_state_dict(torch.load(args.NET_G_path))
        print("Generator loaded from: ", args.NET_G_path)
    elif args.STAGE1_G_path != "":
        netG.stage1_gen.load_state_dict(torch.load(args.STAGE1_G_path))
        print("Generator 1 loaded from: ", args.STAGE1_G_path)
    else:
        print("Please give the Stage 1 generator path")
        return
    
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


def run(args):
    if args.STAGE == 1:
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

    if args.embedding_type == "bert":
        training_set = dataset.CUBDataset(pickl_file=args.train_filenames, img_dir=args.images_dir, bert_emb=args.bert_annotations_dir, stage=args.STAGE)
        testing_set = dataset.CUBDataset(pickl_file=args.test_filenames, img_dir=args.images_dir, bert_emb=args.bert_annotations_dir, stage=args.STAGE)
    else:
        training_set = dataset.CUBDataset(pickl_file=args.train_filenames, img_dir=args.images_dir, cnn_emb=args.cnn_annotations_emb_train, stage=args.STAGE)
        testing_set = dataset.CUBDataset(pickl_file=args.test_filenames, img_dir=args.images_dir, cnn_emb=args.cnn_annotations_emb_test, stage=args.STAGE)
    train_data_loader = torch.utils.data.DataLoader(training_set, batch_size=args.train_bs, num_workers=args.train_workers)
    test_data_loader = torch.utils.data.DataLoader(testing_set, batch_size=args.test_bs, num_workers=args.test_workers)
    # util.check_dataset(training_set)
    # util.check_dataset(testing_set)


    # best_accuracy = 0

    util.make_dir(args.image_save_dir)
    util.make_dir(args.model_dir)
    util.make_dir(args.log_dir)
    summary_writer = FileWriter(args.log_dir)

    for epoch in range(1, args.TRAIN_MAX_EPOCH+1):
        print("__"*80)
        start_t = time.time()

        if epoch % lr_decay_step == 0 and epoch > 0:
            gen_lr *= 0.5
            for param_group in optimizerG.param_groups:
                param_group["lr"] = gen_lr
            disc_lr *= 0.5
            for param_group in optimizerD.param_groups:
                param_group["lr"] = disc_lr
        
        errD, errD_real, errD_wrong, errD_fake, errG, kl_loss, count = engine.train_new_fn(
            train_data_loader, args, netG, netD, real_labels, fake_labels, 
            noise, fixed_noise,  optimizerD, optimizerG, epoch, count, summary_writer)
        
        end_t = time.time()
        
        print(f"[{epoch}/{args.TRAIN_MAX_EPOCH}] Loss_D: {errD:.4f}, Loss_G: {errG:.4f}, Loss_KL: {kl_loss:.4f}, Loss_real: {errD_real:.4f}, Loss_wrong: {errD_wrong:.4f}, Loss_fake: {errD_fake:.4f}, Total Time: {end_t-start_t :.2f} sec")
        if epoch % args.TRAIN_SNAPSHOT_INTERVAL == 0 or epoch == 1:
            util.save_model(netG, netD, epoch, args)
    
    util.save_model(netG, netD, args.TRAIN_MAX_EPOCH, args)
    summary_writer.close()

 
def sample(args, datapath):
    if args.STAGE == 1:
        netG, _ = load_stage1(args)
    else:
        netG, _ = load_stage2(args)
    netG.eval()

    ###* Load text embeddings generated from the encoder:
    t_file = torchfile.load(datapath)
    captions_list = t_file.raw_txt
    embeddings = np.concatenate(t_file.fea_txt, axis=0)
    num_embeddings = len(captions_list)
    print(f"Successfully load sentences from: {args.datapath}")
    print(f"Total number of sentences: {num_embeddings}")
    print(f"Num embeddings: {num_embeddings} {embeddings.shape}")

    ###* Path to save generated samples:
    save_dir = args.NET_G[:args.NET_G.find(".pth")]
    util.make_dir(save_dir)

    batch_size = np.minimum(num_embeddings, args.train_bs)
    nz = args.n_z
    noise = Variable(torch.FloatTensor(batch_size, nz))
    noise = noise.to(args.device)
    count = 0
    while count < num_embeddings:
        if count > 3000:
            break
        iend = count + batch_size
        if iend > num_embeddings:
            iend = num_embeddings
            count = num_embeddings - batch_size
        embeddings_batch = embeddings[count:iend]
        # captions_batch = captions_list[count:iend]
        text_embedding = Variable(torch.FloatTensor(embeddings_batch))
        text_embedding = text_embedding.to(args.device)

        ###* Generate fake images:
        noise.data.normal_(0, 1)
        _, fake_imgs, mu, logvar = netG(text_embedding, noise)
        for i in range(batch_size):
            save_name = f"{save_dir}/{count+i}.png"  
            im = fake_imgs[i].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            # print("im", im.shape)
            im = np.transpose(im, (1, 2, 0))
            # print("im", im.shape)
            im = Image.fromarray(im)
            im.save(save_name)
        count += batch_size
    
if __name__ == "__main__":
    args_ = args.get_all_args()
    args.print_args(args_)
    run(args_)
    # datapath = os.path.join(args_.datapath, "test/val_captions.t7")
    # sample(args_, datapath)
