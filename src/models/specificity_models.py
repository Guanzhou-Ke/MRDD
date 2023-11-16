import math

import torch
from torch import nn
from torch.nn import init

from .autoencoder import Encoder, Decoder
from .resnet import resnet18, resnet34, resnet50


class ViewSpecificAE(nn.Module):
    
    def __init__(self, 
                 c_dim=10, 
                 c_enable=True,
                 v_dim=15, 
                 latent_ch=10, 
                 num_res_blocks=3,
                 block_size=8,
                 channels=1, 
                 basic_hidden_dim=16,
                 ch_mult=[1,2,4,8],
                 kld_weight=0.00025,
                 init_method='kaiming',
                 device='cpu') -> None:
        super().__init__()
        
        # view-specific id.
        self.latent_ch = latent_ch
        self.device = device
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        
        self.block_size = block_size
        
        self.basic_hidden_dim = basic_hidden_dim
        self.v_dim = v_dim
        self.c_dim = c_dim
        self.c_enable = c_enable
        self.input_channel = channels
        self.kld_weight = kld_weight
        self.build_encoder_and_decoder()
               
        self.recons_criterion = nn.MSELoss(reduction='sum')
        # self.recons_criterion = nn.MSELoss()
        # self.apply(self.weights_init(init_type=init_method))
        
        
    def weights_init(self, init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                # print m.__class__.__name__
                if init_type == 'gaussian':
                    init.normal_(m.weight, 0.0, 0.02)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0.0)

        return init_fun
                
        
    def build_encoder_and_decoder(self):
        # self._encoder = resnet18(pretrained=False, in_channel=self.input_channel, output_layer=6)
        self._encoder = Encoder(hidden_dim=self.basic_hidden_dim, 
                                in_channels=self.input_channel, 
                                z_channels=self.latent_ch, 
                                ch_mult=self.ch_mult, 
                                num_res_blocks=self.num_res_blocks, 
                                resolution=1, 
                                use_attn=False, 
                                attn_resolutions=None,
                                double_z=False)
        self._decoder = Decoder(hidden_dim=self.basic_hidden_dim, 
                                out_channels=self.input_channel, 
                                in_channels=self.latent_ch, 
                                z_channels=self.latent_ch, 
                                ch_mult=self.ch_mult,
                                num_res_blocks=self.num_res_blocks, 
                                resolution=1, 
                                use_attn=False, 
                                attn_resolutions=None,
                                double_z=False)
        
        self.to_dist_layer = nn.Linear(self.latent_ch * (self.block_size **2), self.v_dim*2)
        if self.c_enable:
            self.to_decoder_input = nn.Linear(self.v_dim+self.c_dim, self.latent_ch * (self.block_size **2))
        else:
            self.to_decoder_input = nn.Linear(self.v_dim, self.latent_ch * (self.block_size **2))
            
    
    def get_encoder_params(self):
        return self._encoder.parameters()
    
    def latent(self, x):
        latent = self._encoder(x)
        latent = torch.flatten(latent, start_dim=1)
        z = self.to_dist_layer(latent)
        mu, logvar = torch.split(z, self.v_dim, dim=1)
        z = self.reparameterize(mu, logvar)
        return z
        
    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        latent = self._encoder(x)
        latent = torch.flatten(latent, start_dim=1) 
        z = self.to_dist_layer(latent)
        mu, logvar = torch.split(z, self.v_dim, dim=1)

        return [mu, logvar]

    def decode(self, z):
        z = self.to_decoder_input(z)
        z = z.view(-1, self.latent_ch, self.block_size, self.block_size)
        result = self._decoder(z)
        return result

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, x, y=None):
        
        mu, logvar = self.encode(x)
        
        z = self.reparameterize(mu, logvar)
        if y is not None:
            z = torch.cat([z, y], dim=1)
            
        return self.decode(z), mu, logvar
    
    
    def get_loss(self, x, y):
        out, mu, logvar = self(x, y)
        
        recons_loss = self.recons_criterion(out, x)
        
        kld_loss = self.kld_weight * (torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0))
        return recons_loss, kld_loss
    
    
    def sample(self, num_samples, y):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param y: (Tensor) controlled labels.
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.v_dim).to(self.device)

        z = torch.cat([z, y], dim=1).to(self.device)
        samples = self.decode(z)
        return samples
    
