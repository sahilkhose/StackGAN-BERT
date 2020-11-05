"""Assortment of layers for use in models.py.
Refer to StackGAN paper: https://arxiv.org/pdf/1612.03242.pdf 
for variable names and working.

Authors:
    Abhiraj Tiwari (abhirajtiwari@gmail.com)
    Sahil Khose (sahilkhose18@gmail.com)
"""
import torch
import torch.nn as nn

from torch.autograd import Variable

class CAug(nn.Module):
    """Module for conditional augmentation.
    Takes input as bert embeddings of annotations and sends output to Stage 1 and 2 generators.
    """
    def __init__ (self, bert_dim=768, n_g=128, device=torch.cuda):
        """
        @param bert_dim (int)           : Size of bert annotation embeddings. 
        @param n_g      (int)           : Dimension of mu, epsilon and c_0_hat
        @param device   (torch.device)  : cuda/cpu
        """
        super(CAug, self).__init__()
        self.bert_dim = bert_dim
        self.n_g = n_g
        self.ff = nn.Linear(self.bert_dim, self.n_g*2)  # To split in mu and sigma
        self.relu = nn.ReLU()
        self.device = device

    def forward(self, x):
        """
        @param   x       (torch.tensor): Text embedding.                 (batch, bert_dim)
        @returns c_0_hat (torch.tensor): Gaussian conditioning variable. (batch, n_g)
        """
        enc = self.relu(self.ff(x))  # (batch, n_g*2)
        mu = enc[:,:self.n_g]  # (batch, n_g)
        sigma = enc[:,self.n_g:]  # (batch, n_g)
        if self.device==torch.cuda:
            epsilon = Variable(torch.cuda.FloatTensor(sigma.size()).normal_())
        else:
            epsilon = Variable(torch.FloatTensor(sigma.size()).normal_())
        c_0_hat = epsilon * sigma + mu  # (batch, n_g)
        return c_0_hat

class Stage1Generator(nn.Module):
    """
    Stage 1 generator.
    Takes in input from Conditional Augmentation and outputs 64x64 image to Stage1Discrimantor.
    """
    def __init__(self, n_g=128, n_z=100):
        """
        @param n_g (int) : Dimension of c_0_hat.
        @param n_z (int) : Dimension of noise vector.
        """
        super(Stage1Generator, self).__init__()
        self.n_g = n_g
        self.n_z = n_z
        self.ff = nn.Linear(self.n_g + self.n_z, (self.n_g*8) * 4*4)

        self.inp_ch = self.n_g*8

        self.up1 = _upsample(self.inp_ch,    self.inp_ch//2)  # (batch, 512, 8, 8)
        self.up2 = _upsample(self.inp_ch//2, self.inp_ch//4)  # (batch, 256, 16, 16)
        self.up3 = _upsample(self.inp_ch//4, self.inp_ch//8)  # (batch, 128, 32, 32)
        self.up4 = _upsample(self.inp_ch//8, self.inp_ch//16) # (batch, 64, 64, 64)

        self.conv_fin = nn.Conv2d(self.inp_ch//16, 3)  # (batch, 3, 64, 64)

    def forward(self, c_0_hat):
        """
        @param   c_0_hat (torch.tensor) : Output of Conditional Augmentation (batch, n_g) 
        @returns out     (torch.tensor) : Generator 1 image output           (batch, 64, 64, 3)
        """
        batch_size = c_0_hat.size()[0]
        inp = torch.cat((c_0_hat, torch.empty((batch_size, self.n_z)).normal_()), dim=1)  # (batch, n_g+n_z; i.e 228)
        inp = self.ff(inp)  # (batch, 1024 * 4 * 4) : 1024 => n_g * 8 => 128 * 8
        inp = inp.reshape((batch_size, self.inp_ch, 4, 4))  # (batch, 1024, 4, 4)
        inp = self.up4(self.up3(self.up2(self.up1(inp))))  # (batch, 64, 64, 64)
        out = self.conv_fin(inp)  # (batch, 3, 64, 64)
        return out

class Stage1Discriminator(nn.Module):
    """
    Stage 1 generator
    """
    def __init__(self, n_d=128, m_d=4, bert_dim=768):
        super(Stage1Discriminator, self).__init__()
        self.n_d = n_d
        self.m_d = m_d
        self.bert_dim = bert_dim

        self.ff = nn.Linear(self.bert_dim, self.n_d)

    def forward(self, text_emb, img):
        batch_size = text_emb.size()[0]
        compressed = self.ff(text_emb)
        compressed = compressed.repeat(batch_size, self.m_d, self.m_d, self.n_d)

def _upsample(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

if __name__ == "__main__":
    pass