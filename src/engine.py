# import config
import util

import torch
import torch.nn as nn

from torch.utils.tensorboard import summary
from torch.utils.tensorboard import FileWriter
from tqdm import tqdm

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def disc_loss(disc, real_imgs, fake_imgs, real_labels, fake_labels, conditional_vector):
    loss_fn = nn.BCELoss()
    batch_size = real_imgs.shape[0]
    cond = conditional_vector.detach()
    fake_imgs = fake_imgs.detach()

    print(cond.shape, real_imgs.shape, real_labels.shape)
    real_loss = loss_fn(disc(cond, real_imgs), real_labels)
    fake_loss = loss_fn(disc(cond, fake_imgs), fake_labels)
    wrong_loss = loss_fn(disc(cond[1:], real_imgs[:-1]), fake_labels[1:])

    loss = real_loss + (fake_loss+wrong_loss)*0.5
    return loss

def gen_loss(disc, fake_imgs, real_labels, conditional_vector):
    loss_fn = nn.BCELoss()
    cond = conditional_vector.detach()
    fake_loss = loss_fn(disc(cond, fake_imgs), real_labels)
    return fake_loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def train_new_fn(data_loader, args, netG, netD, real_labels, fake_labels, noise, fixed_noise,  optimizerD, optimizerG, epoch, count, summary_writer):
    for batch_id, data in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Train Epoch {epoch}/{args.TRAIN_MAX_EPOCH}"):
        ###* Prepare training data:
        text_emb, real_images = data
        text_emb = text_emb.to(args.device)
        real_images = real_images.to(args.device)

        ###* Generate fake images:
        noise.data.normal_(0, 1)
        _, fake_images, mu, logvar = netG(text_emb, noise)

        ###* Update D network:
        netD.zero_grad()
        errD, errD_real, errD_wrong, errD_fake = disc_loss(netD, real_images, fake_images, real_labels, fake_labels, text_emb)
        errD.backward()
        optimizerD.step()

        ###* Update G network:
        netG.zero_grad()
        errG = gen_loss(netD, fake_images, real_labels, text_emb)
        kl_loss = KL_loss(mu, logvar)
        errG_total = errG + kl_loss * args.TRAIN_COEFF_KL
        errG_total.backward()
        optimizerG.step()

        count += 1

        if batch_id % 100 == 0:
            summary_D = summary.scalar("D_loss", errD.data[0])
            summary_D_r = summary.scalar("D_loss_real", errD_real.data[0])
            summary_D_w = summary.scalar("D_loss_wrong", errD_wrong.data[0])
            summary_D_f = summary.scalar("D_loss_fake", errD_fake.data[0])
            summary_G = summary.scalar("G_loss", errG.data[0])
            summary_KL = summary.scalar("KL_loss", kl_loss.data[0])

            summary_writer.add_summary(summary_D, count)
            summary_writer.add_summary(summary_D_r, count)
            summary_writer.add_summary(summary_D_w, count)
            summary_writer.add_summary(summary_D_f, count)
            summary_writer.add_summary(summary_G, count)
            summary_writer.add_summary(summary_KL, count)
            
            ###* save the image result for each epoch:
            lr_fake, fake, _, _ = netG(text_emb, fixed_noise)
            util.save_img_results(real_images, fake, epoch, args.image_save_dir)
            if lr_fake is not None:
                util.save_img_results(None, lr_fake, epoch, args.image_save_dir)



def train_fn(data_loader, Discriminator, Generator, optimD, optimG, device, epoch): 
    Discriminator.train()
    Generator.train()
    #LOSS = 0.
    DISC_LOSS = 0.
    GEN_LOSS = 0.

    for batch_id, data in tqdm(enumerate(data_loader), total=len(data_loader)): #, desc=f"Train Epoch {epoch}/{config.EPOCHS}"):
        text_embs, images = data
        images = torch.tensor(images)

        # Loading it to device
        text_embs = text_embs.to(device, dtype=torch.float)
        images = images.to(device, dtype=torch.float)

        # print(text_embs.size())
        # ROUGH TRAIN
        batch_size = text_embs.shape[0]
        # print("batch_size:", batch_size)
        n_z = 100
        noise = torch.empty((batch_size, n_z)).normal_()
        _, fake, mu, logvar = Generator(text_embs, noise.to(device, dtype=torch.float))

        real_labels = torch.ones(batch_size)
        fake_labels = torch.zeros(batch_size)

        # TODO create real_labels, fake_labels, d1 takes as input text_emb as input, how do we pass mu :/
        Discriminator.zero_grad()
        loss_disc = disc_loss(Discriminator, images, fake, real_labels, fake_labels, text_embs) # XXX
        DISC_LOSS += loss_disc.detach()
        loss_disc.backward()
        optimD.step()

        Generator.zero_grad()
        loss_gen = gen_loss(Discriminator, fake, real_labels, text_embs)
        kl = KL_loss(mu, logvar)
        generator_loss = loss_gen + 1*kl # TODO add this arg, this the lambda in the paper
        GEN_LOSS += generator_loss.detach()
        generator_loss.backward()
        optimG.step()

        break

    DISC_LOSS /= len(data_loader) # XXX why
    GEN_LOSS /= len(data_loader)
    return DISC_LOSS, GEN_LOSS # XXX why


def eval_fn(data_loader, model, device, epoch):
    model.eval()
    fin_y = []
    fin_outputs = []
    LOSS = 0.

    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            text_embs, images = data

            # Loading it to device
            text_embs = text_embs.to(device, dtype=torch.float)
            images = images.to(device, dtype=torch.float)
            
            # getting outputs from model and calculating loss
            outputs = model(text_embs, images)
            loss = loss_fn(outputs, images) # TODO figure this out
            LOSS += loss

            # for calculating accuracy and other metrics # TODO figure this out
            fin_y.extend(images.view(-1, 1).cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    LOSS /= len(data_loader)
    return fin_outputs, fin_y, LOSS
