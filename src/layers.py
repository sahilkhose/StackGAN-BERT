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
        self.ff = nn.Linear(self.n_g + self.n_z, (self.n_g*8) * 4*4)  # (batch, 1024 * (4*4))

        self.inp_ch = self.n_g*8  
        """
        To map (4, 4) -> (64, 64)
        There will be 4 -> 8, 8 -> 16, 16 -> 32, 32 -> 64 : 4 upsampling blocks
        The hidden dimension halves with every upsampling block.
        """

        self.up1 = _upsample(self.inp_ch,    self.inp_ch//2)  # (batch, 512, 8, 8)
        self.up2 = _upsample(self.inp_ch//2, self.inp_ch//4)  # (batch, 256, 16, 16)
        self.up3 = _upsample(self.inp_ch//4, self.inp_ch//8)  # (batch, 128, 32, 32)
        self.up4 = _upsample(self.inp_ch//8, self.inp_ch//16) # (batch, 64, 64, 64)

        self.conv_fin = nn.Conv2d(self.inp_ch//16, 3, kernel_size=3, padding=1)  # (batch, 3, 64, 64)

        #? There is no mention of conv2d in the paper
        #? They mention using upsampling blocks and the last upsampling block does not have
        #? a batch norm and relu activation
        #? We can just set output channels of self.up4 to 3 to solve this problem
        #? I think this conv2d is used because of the keras implementation

    def forward(self, c_0_hat):
        """
        @param   c_0_hat (torch.tensor) : Output of Conditional Augmentation (batch, n_g) 
        @returns out     (torch.tensor) : Generator 1 image output           (batch, 3, 64, 64)
        """
        batch_size = c_0_hat.size()[0]
        # Concat c_0_hat with z:
        inp = torch.cat((c_0_hat, torch.empty((batch_size, self.n_z)).normal_()), dim=1)  # (batch, n_g + n_z) (batch, 128 + 100)

        inp = self.ff(inp)  # (batch, 1024 * 4 * 4) : 1024 => n_g * 8 => 128 * 8
        
        inp = inp.reshape((batch_size, self.inp_ch, 4, 4))  # (batch, 1024, 4, 4)
        inp = self.up4(self.up3(self.up2(self.up1(inp))))  # (batch, 64, 64, 64)
        out = self.conv_fin(inp)  # (batch, 3, 64, 64)
        return out

class Stage1Discriminator(nn.Module):
    """
    Stage 1 discriminator
    """
    def __init__(self, n_d=128, m_d=4, bert_dim=768, img_dim=64):
        super(Stage1Discriminator, self).__init__()
        self.n_d = n_d
        self.m_d = m_d
        self.bert_dim = bert_dim
        lr_slope = 0.01

        self.ff_for_text = nn.Linear(self.bert_dim, self.n_d)
        self.down_sample = nn.Sequential(
            nn.Conv2d(3, img_dim, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(lr_slope, inplace=True), # TODO change slope?

            _downsample(img_dim, img_dim*2),
            _downsample(img_dim*2, img_dim*4),
            _downsample(img_dim*4, img_dim*8)
        )
        self.conv1x1 = nn.Conv2d(img_dim*8+self.n_d, img_dim*8+self.n_d, kernel_size=1)
        self.final = nn.Linear(self.m_d*self.m_d*(self.n_d+(img_dim*8)),1)
        self.sig = nn.Sigmoid()

    def forward(self, text_emb, img):
        batch_size = img.size()[0]

        # image encode
        enc = self.down_sample(img)
        # text emb
        compressed = self.ff_for_text(text_emb)
        compressed = compressed[:,:,None,None].repeat(1, 1, self.m_d, self.m_d)

        con = torch.cat((enc, compressed), dim=1)
        con = self.conv1x1(con)
        return self.sig(self.final(con.flatten(start_dim=1)))



class Stage2Generator(nn.Module):
    """
    Stage 2 generator.
    Takes in input from Conditional Augmentation and outputs 256x256 image to Stage2Discrimantor.
    """
    def __init__(self, n_g=128):
        """
        @param n_g (int) : Dimension of c_0_hat.
        """
        super(Stage2Generator, self).__init__()
        self.n_g = n_g
        self.down1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)  # (batch, 128, 64, 64)
        self.relu1 = nn.LeakyReLU()
        self.down2 = _downsample(128, 256)  # (batch, 256, 32, 32)
        self.down3 = _downsample(256, 512)  # (batch, 512, 16, 16)
        self.res1 = _residual(512 + self.n_g, 512 + self.n_g)  # (batch, 640, 16, 16)
        self.res2 = _residual(512 + self.n_g, 512 + self.n_g)  # (batch, 640, 16, 16)
        self.res3 = _residual(512 + self.n_g, 512 + self.n_g)  # (batch, 640, 16, 16)
        self.res4 = _residual(512 + self.n_g, 512 + self.n_g)  # (batch, 640, 16, 16)
        self.up1 = _upsample(512 + self.n_g, 512)  # (batch, 512, 32, 32)
        self.up2 = _upsample(512, 256)  # (batch, 256, 64, 64)
        self.up3 = _upsample(256, 128)  # (batch, 128, 128, 128)
        self.up4 = _upsample(128, 64)   # (batch, 64, 256, 256)
        self.conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # (batch, 3, 256, 256)

    def forward(self, c_0_hat, s1_image):
        """
        @param   c_0_hat  (torch.tensor) : Output of Conditional Augmentation (batch, n_g) 
        @param   s1_image (torch.tensor) : Ouput of Stage 1 Generator         (batch, 3, 64, 64)
        @returns out      (torch.tensor) : Generator 2 image output           (batch, 3, 256, 256)
        """
        batch_size = c_0_hat.size()[0]

        # downsample:
        down_out = self.down3(self.down2(self.relu1(self.down1(s1_image))))
        c_out = c_0_hat.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)  # (batch, n_g, 16, 16)  # (batch, 128, 16, 16)
        # (batch, n_g) -> (batch, n_g, 1, 1) -> (batch, n_g, 16, 16) : n_g = 128
        concat_out = torch.cat((down_out, c_out), dim=1)  # (batch, n_g + 512, 16, 16) # (batch, 640, 16, 16)

        res1_out = self.res1(concat_out)  # (batch, 640, 16, 16)

        res2_in = res1_out + concat_out
        res2_out = self.res2(res2_in)  # (batch, 640, 16, 16)

        res3_in = res2_out + res2_in
        res3_out = self.res3(res3_in)  # (batch, 640, 16, 16)

        res4_in = res3_out + res3_in
        res4_out = self.res4(res4_in)  # (batch, 640, 16, 16)

        gen_out = self.conv(self.up4(self.up3(self.up2(self.up1(res4_out)))))  # (batch, 3, 256, 256)

        return gen_out


class Stage2Discriminator(nn.Module):
    """
    Stage 2 discriminator
    """

    def __init__(self, n_d=128, m_d=4, bert_dim=768, img_dim=256):
        super(Stage2Discriminator, self).__init__()
        self.n_d = n_d
        self.m_d = m_d
        self.bert_dim = bert_dim
        lr_slope = 0.01

        self.ff_for_text = nn.Linear(self.bert_dim, self.n_d)
        self.down_sample = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=True),  # (batch, 16, 128, 128)
            nn.LeakyReLU(lr_slope, inplace=True),  # TODO change slope?
            _downsample(16, 32),  # (batch, 32, 64, 64)
            _downsample(32, 64),  # (batch, 64, 32, 32)
            _downsample(64, 128), # (batch, 128, 16, 16)
            _downsample(128, 256),# (batch, 256, 8, 8)
            _downsample(256, 512),# (batch, 512, 4, 4)
        )
        self.conv1x1 = nn.Conv2d(img_dim*2+self.n_d, img_dim*2+self.n_d, kernel_size=1)
        self.final = nn.Linear(self.m_d*self.m_d*(self.n_d+(img_dim*2)), 1)
        self.sig = nn.Sigmoid()

    def forward(self, text_emb, img):
        batch_size = img.size()[0]

        # image encode
        enc = self.down_sample(img)
        # text emb
        compressed = self.ff_for_text(text_emb)
        compressed = compressed[:, :, None, None].repeat(
            1, 1, self.m_d, self.m_d)

        con = torch.cat((enc, compressed), dim=1)
        con = self.conv1x1(con)
        return self.sig(self.final(con.flatten(start_dim=1)))


def _residual(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def _downsample(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    #TODO add layer_1 boolean argument with if else conditions
    #TODO figure out leaky relu 
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )

def _upsample(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

if __name__ == "__main__":
    batch_size = 2
    emb = torch.randn((batch_size, 768))
    ca1 = CAug(768,128,'cpu')
    generator1 = Stage1Generator()
    ca2 = CAug(768,128,'cpu')
    generator2 = Stage2Generator()

    discriminator1 = Stage1Discriminator()
    discriminator2 = Stage2Discriminator()


    out_ca1 = ca1(emb)
    print("ca1 output size: ", out_ca1.size())  # (batch_size, 128)
    assert out_ca1.shape == (batch_size, 128)
    gen1 = generator1(out_ca1) 
    print("output1 image dimensions :", gen1.size())  # (batch_size, 3, 64, 64)
    assert gen1.shape == (batch_size, 3, 64, 64)
    print()

    disc1 = discriminator1(emb, gen1)
    print("output1 discriminator", disc1.size())  # (batch_size, 1)
    assert disc1.shape == (batch_size, 1)
    print()

    out_ca2 = ca2(emb)
    print("ca2 output size: ", out_ca2.size())  # (batch_size, 128)
    assert out_ca2.shape == (batch_size, 128)
    gen2 = generator2(out_ca2, gen1)
    print("output2 image dimensions :", gen2.size())  # (batch_size, 3, 256, 256)
    assert gen2.shape == (batch_size, 3, 256, 256)
    print()

    disc2 = discriminator2(emb, gen2)
    print("output2 discriminator", disc2.size())  # (batch_size, 1)
    assert disc2.shape == (batch_size, 1)
