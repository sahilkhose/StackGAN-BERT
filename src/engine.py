# import config
import torch
import torch.nn as nn
from tqdm import tqdm

# TODO following is very basic {train, eval}_fn and random loss_fn. Figure out for GANs

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def disc_loss(disc, real_imgs, fake_imgs, real_labels, fake_labels, conditional_vector):
    loss_fn = nn.BCELoss()
    cond = conditional_vector.detach()
    fake_imgs = fake_imgs.detach()

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


def train_fn(data_loader, Discriminator, Generator, device, epoch): 
    Discriminator.train()
    Generator.train()
    #LOSS = 0.
    DISC_LOSS = 0.
    GEN_LOSS = 0.
    lr = 0.0002 # TODO decay this every 100 epochs by 0.5
    optimG = torch.optim.Adam(Generator.parameters(), lr, betas=(0.5, 0.999)) # Beta value from github impl
    optimD = torch.optim.Adam(Discriminator.parameters(), lr, betas=(0.5, 0.999))

    for batch_id, data in tqdm(enumerate(data_loader), total=len(data_loader)): #, desc=f"Train Epoch {epoch}/{config.EPOCHS}"):
        text_embs, images = data
        images = torch.tensor(images)

        # Loading it to device
        text_embs = text_embs.to(device, dtype=torch.float)
        images = images.to(device, dtype=torch.float)

        # ROUGH TRAIN
         # TODO create noise here
        # noise  = torch.Tensor()
        batch_size = text_embs.shape[0]
        print("batch_size:", batch_size)
        n_z = 100
        noise = torch.empty((batch_size, n_z)).normal_()
        _, fake, mu, logvar = Generator(text_embs, noise.to(device, dtype=torch.float))

         # TODO create real_labels, fake_labels, d1 takes as input text_emb as input, how do we pass mu :/
        # Discriminator.zero_grad()
        # loss_disc = disc_loss(Discriminator, images, fake, real_labels, fake_labels, mu)
        # DISC_LOSS += loss_disc.detach()
        # loss_disc.backward()
        # optimD.step()

        # Generator.zero_grad()
        # loss_gen = gen_loss(Discriminator, fake, real_labels, mu)
        # kl = KL_loss(mu, logvar)
        # generator_loss = loss_gen + args.kl_hyperparam*kl # TODO add this arg, this the lambda in the paper
        # GEN_LOSS += generator_loss.detach()
        # generator_loss.backward()
        # optimG.step()

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
