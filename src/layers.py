import torch
import torch.nn as nn
from torch.autograd import Variable

class CAug(nn.Module):
    def __init__ (self, bert_dim=768, n_g=128, device=torch.cuda):
        """
        Module for conditional augmentation
        """
        super(CAug, self).__init__()
        self.bert_dim = bert_dim
        self.n_g = n_g # This is the dimention of mu
        self.ff = nn.Linear(self.bert_dim, self.n_g*2)
        self.relu = nn.ReLU()
        self.device = device
    def forward(self, x):
        """
        input:
            @param x: text embedding (batch, bert_dim)
        output:
            @returns c_0: refer StackGAN paper (batch, n_g)
        """
        enc = self.relu(self.ff(x))
        mu = enc[:,:self.n_g]
        sig = enc[:,self.n_g:]
        if self.device==torch.cuda:
            epsilon = Variable(torch.cuda.FloatTensor(sig.size()).normal_())
        else:
            epsilon = Variable(torch.FloatTensor(sig.size()).normal_())
        c_0 = epsilon * sig + mu
        return c_0

class Stage1Generator(nn.Module):
    """
    Stage 1 generator
    """
    def __init__(self, n_g=128, n_z=100):
        super(Stage1Generator, self).__init__()
        self.n_g = n_g
        self.n_z = n_z
        self.ff = nn.Linear(self.n_g+self.n_z, (self.n_g*8)*4*4)

        self.inp_ch = self.n_g*8

        self.up1 = _upsample(self.inp_ch, self.inp_ch//2)
        self.up2 = _upsample(self.inp_ch//2, self.inp_ch//4)
        self.up3 = _upsample(self.inp_ch//4, self.inp_ch//8)
        self.up4 = _upsample(self.inp_ch//8, self.inp_ch//16)

        self.conv_fin = nn.Conv2d(self.inp_ch//16, 3)

    def forward(self, c_0): 
        batch_size = c_0.size()[0]
        inp = torch.cat((c_0, torch.empty((batch_size, self.n_z)).normal_())) #(batch, n_g+n_z; i.e 228)
        inp = self.ff(inp)
        inp = inp.reshape((batch_size, self.inp_ch, 4, 4))
        inp = self.up4(self.up3(self.up2(self.up1(inp))))
        out = self.conv_fin(inp)
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
