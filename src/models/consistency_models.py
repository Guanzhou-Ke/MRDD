import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.distributions as dist

# import sys
# sys.path.append('/home/xyzhang/guanzhouke/cvpr24/src/models')
# from autoencoder import Encoder, Decoder
from .autoencoder import Encoder, Decoder
from .resnet import resnet18, resnet34, resnet50
from utils import mask_image

class ConsistencyAE(nn.Module):

    def __init__(self,
                 basic_hidden_dim=32, 
                 c_dim=64, 
                 continous=True, 
                 in_channel=3,
                 num_res_blocks=3, 
                 ch_mult=[1, 2, 4, 8],
                 block_size=8,
                 latent_ch=10,
                 temperature=1.0,
                 kld_weight=0.00025,
                 views=2,
                 alpha=0.5,
                 categorical_dim=10) -> None:
        """
        """
        super().__init__()

        self.c_dim = c_dim
        self.continous = continous
        self.in_channel = in_channel
        self.ch_mult = ch_mult
        self.block_size = block_size
        self.basic_hidden_dim = basic_hidden_dim
        self.num_res_blocks = num_res_blocks
        self.latent_ch = latent_ch
        self.anneal_rate = 0.00003
        self.min_temp = 0.5
        self.temp = temperature
        self.views = views
        self.kld_weight = kld_weight
        self.categorical_dim = categorical_dim
        self.alpha = alpha
        
        
        self._encoder = Encoder(hidden_dim=self.basic_hidden_dim, 
                                in_channels=self.in_channel, 
                                z_channels=self.latent_ch, 
                                ch_mult=self.ch_mult, 
                                num_res_blocks=self.num_res_blocks, 
                                resolution=1, 
                                use_attn=False, 
                                attn_resolutions=None,
                                double_z=False)
        # self._encoder = resnet18(pretrained=False, in_channel=self.in_channel, output_layer=6)
    
        self.decoders = nn.ModuleList([Decoder(hidden_dim=self.basic_hidden_dim, 
                                out_channels=self.in_channel, 
                                in_channels=self.latent_ch, 
                                z_channels=self.latent_ch, 
                                ch_mult=self.ch_mult,
                                num_res_blocks=self.num_res_blocks, 
                                resolution=1, 
                                use_attn=False, 
                                attn_resolutions=None,
                                double_z=False) for _ in range(self.views)])
        
        if self.continous:
            # continous code.
            self.fc_z = nn.Linear(512 * self.views, self.c_dim*2)
            # self.fc_z = nn.Linear(2048 * self.views, self.c_dim*2)
            
            # self.fc_z = nn.Linear(self.latent_ch * self.block_size ** 2 * self.views, self.c_dim*2)
            self.to_decoder_input = nn.Linear(self.c_dim, self.latent_ch * self.block_size **2)
        else:
            # discrete code.
            self.fc_z = nn.Linear(self.latent_ch * self.block_size ** 2, self.c_dim * self.categorical_dim)
            self.to_decoder_input = nn.Linear(self.c_dim * self.categorical_dim, self.latent_ch * self.block_size **2)
            
    def forward(self, Xs):
    
        if self.continous:
            mu, logvar = self.encode(Xs)
            z = self.cont_reparameterize(mu, logvar)
        else:
            beta = self.encode(Xs)
            z = self.disc_reparameterize(beta)    
        
        recons = self.decode(z)
        return recons, z
    
    
    def cont_reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu
    
    
    def disc_reparameterize(self, z, eps=1e-7):
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param z: (Tensor) Latent Codes [B x D x Q]
        :return: (Tensor) [B x D]
        """
        # if self.training:
            # Sample from Gumbel
        u = torch.rand_like(z)
        g = - torch.log(- torch.log(u + eps) + eps)
        # Gumbel-Softmax sample
        s = F.softmax((z + g) / self.temp, dim=-1)
        s = s.view(-1, self.c_dim * self.categorical_dim)
        return s
    
        
    def encode(self, Xs):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        latents = []
        for x in Xs:
            latent = self._encoder(x)
            latent = torch.flatten(latent, start_dim=1) 
            latents.append(latent)
        latent = torch.cat(latents, dim=-1) 
        z = self.fc_z(latent)
        if self.continous:
            mu, logvar = torch.split(z, self.c_dim, dim=1)
            return mu, logvar
        else:
            return z.view(-1, self.c_dim, self.categorical_dim)

    def decode(self, z):
        z = self.to_decoder_input(z)
        z = z.view(-1, self.latent_ch, self.block_size, self.block_size)
        return [dec(z) for dec in self.decoders]
        
    
    
    def get_loss(self, Xs, epoch, mask_ratio, mask_patch_size):
        # Masked cross-view distribution modeling.
        Xs_masked = [mask_image(x, mask_patch_size, mask_ratio=mask_ratio) for x in Xs]
        if self.continous:
            mu, logvar = self.encode(Xs_masked)
            kld_loss = self.con_loss(mu, logvar)
    
            z = self.cont_reparameterize(mu, logvar)
        else:
            beta = self.encode(Xs_masked)
            kld_loss = self.disc_loss(beta, epoch)
            
            z = self.disc_reparameterize(beta)
            
        recons = self.decode(z)
        recon_loss = 0.
        return_details = {}
        for v, (x, recon) in enumerate(zip(Xs, recons)):
            sub_loss = F.mse_loss(x, recon, reduction='sum')
            return_details[f'v{v+1}-loss'] = sub_loss.item()
            recon_loss += sub_loss
       
        loss = self.alpha * recon_loss + kld_loss
        
        return_details['total_loss'] = loss.item()
        return_details['recon_loss'] = recon_loss.item()
        return_details['kld_loss'] = kld_loss.item()
        return_details['temparature'] = self.temp
        
        return loss, return_details
    
    
    def con_loss(self, mu, log_var):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return self.kld_weight * kld_loss

        
    def disc_loss(self, Q, epoch) -> dict:
        """"
        Computes the discreate-VAE loss function.
        """
        
        B, N, K = Q.shape
        Q = Q.view(B*N, K)
        q = dist.Categorical(logits=Q)
        p = dist.Categorical(probs=torch.full((B*N, K), 1.0/K).to(Q.device)) # uniform bunch of K-class categorical distributions
        kl = dist.kl.kl_divergence(q, p).view(B, N) # kl is of shape [B*N]
        
        if epoch % 5 == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(-self.anneal_rate * epoch),
                                   self.min_temp)
        
        return torch.mean(torch.sum(kl, dim=1))

    
    def consistency_features(self, Xs):
        if self.continous:
            mu, logvar = self.encode(Xs)
            z = self.cont_reparameterize(mu, logvar)
        return z
        # zs = []
        # for x in Xs:
        #     if self.continous:
        #         mu, logvar = self.encode(x)
        #         z = self.cont_reparameterize(mu, logvar)
        #     else:
        #         beta = self.encode(x)
        #         z = self.disc_reparameterize(beta)  
        #     zs.append(z.unsqueeze(1))
        # zs = torch.cat(zs, dim=1).mean(dim=1)
        # return zs
    
    def sampling(self, samples_num, device='cpu', return_code=False):
        if self.continous:
            z = torch.randn(samples_num, self.c_dim).to(device)
        else:    
            Q = torch.randn((samples_num, self.c_dim, self.categorical_dim))
            z = self.disc_reparameterize(Q).to(device)
        if return_code:
            return z
        else:
            samples = self.decode(z)
            return samples
            

if __name__ == '__main__':
    pass