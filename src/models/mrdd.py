import torch
from torch import nn

from .consistency_models import ConsistencyAE
from .specificity_models import ViewSpecificAE
from .mi_estimators import CLUBSample


class MRDD(nn.Module):

    def __init__(self, args, consistency_encoder_path=None, device='cpu') -> None:
        super().__init__()
        self.args = args
        self.views = self.args.views
        self.device = device

        # consistency encoder.
        if self.args.consistency.enable:
            self.consis_enc = ConsistencyAE(basic_hidden_dim=self.args.consistency.basic_hidden_dim,
                                            c_dim=self.args.consistency.c_dim,
                                            continous=self.args.consistency.continous,
                                            in_channel=self.args.consistency.in_channel,
                                            num_res_blocks=self.args.consistency.num_res_blocks,
                                            ch_mult=self.args.consistency.ch_mult,
                                            block_size=self.args.consistency.block_size,
                                            temperature=self.args.consistency.temperature,
                                            latent_ch=self.args.consistency.latent_ch,
                                            kld_weight=self.args.consistency.kld_weight,
                                            views=self.args.views,
                                            categorical_dim=self.args.dataset.class_num
                                            )
            self.consis_enc.eval()
            if consistency_encoder_path is not None:
                self.consis_enc.load_state_dict(torch.load(
                    consistency_encoder_path, map_location='cpu'), strict=False)
                # freeze consistency network.
                for param in self.consis_enc.parameters():
                    param.requires_grad = False
            
            self.c_dim = self.args.consistency.c_dim
            self.v_dim = self.args.vspecific.v_dim
            
    
        if self.args.vspecific.enable:
            # create view-specific encoder.
            for i in range(self.args.views):
                self.__setattr__(f"venc_{i+1}", ViewSpecificAE(c_dim=self.args.consistency.c_dim, 
                                                            c_enable=self.args.consistency.enable,
                                                            v_dim=self.v_dim, 
                                                            latent_ch=self.args.vspecific.latent_ch, 
                                                            num_res_blocks=self.args.vspecific.num_res_blocks,
                                                            block_size=self.args.vspecific.block_size,
                                                            channels=self.args.consistency.in_channel, 
                                                            basic_hidden_dim=self.args.vspecific.basic_hidden_dim,
                                                            ch_mult=self.args.vspecific.ch_mult,
                                                            init_method=self.args.backbone.init_method,
                                                            kld_weight=self.args.vspecific.kld_weight,
                                                            device=self.device))
                if self.args.consistency.enable:
                    self.__setattr__(f"mi_est_{i+1}", CLUBSample(self.c_dim, self.v_dim, hidden_size=self.args.disent.hidden_size))
                

        # Common feature pooling method. mean, sum, or first
        self.pooling_method = self.args.fusion.pooling_method

    def get_disentangling_params(self, vid):
        params = [self.__getattr__(f"venc_{vid+1}").get_encoder_params(),
                  self.__getattr__(f"mi_est_{vid+1}").parameters()]
        return params

    def get_reconstruction_params(self, vid):
        params = self.__getattr__(f"venc_{vid+1}").parameters()
        return params
    
    def get_vsepcific_params(self, vid):
        params = [self.__getattr__(f"venc_{vid+1}").parameters(),
                  self.__getattr__(f"mi_est_{vid+1}").parameters()]
        return params
    
    
    def forward(self, Xs):
        return self.enc_dec(Xs)
        


    def sampling(self, samples_nums, device='cpu'):
        """
        samples_num: e
        """
        C = self.consis_enc.sampling(samples_nums // 2, device, return_code=True)
        outs = []
        for b in range(samples_nums):
            C = torch.cat([C[b, :].unsqueeze(0)]*samples_nums, dim=0)
            for i in range(self.args.views):
                venc = self.__getattr__(f"venc_{i+1}")
                out = venc.sample(samples_nums, C)
                outs.append(out)
        return torch.cat(outs) 

    def get_disentangling_loss(self, Xs):
        tot_loss = []
        
        if self.args.consistency.enable:
            with torch.no_grad():
                C = self.consis_enc.consistency_features(Xs)
        else:
            C = None
    
        for i in range(self.views):
            venc = self.__getattr__(f"venc_{i+1}")
            mi_est = self.__getattr__(f"mi_est_{i+1}")

            mi_lb = mi_est.learning_loss(venc.latent(Xs[i]), C)
            disent_loss = self.args.disent.lam * mi_lb
            
            tot_loss.append(disent_loss)
            
        return tot_loss
    
    def get_reconstruction_loss(self, Xs):
        tot_loss = []
        
        if self.args.consistency.enable:
            with torch.no_grad():
                C = self.consis_enc.consistency_features(Xs)
        else:
            C = None
           
        for i in range(self.views):
            venc = self.__getattr__(f"venc_{i+1}")
            
            recons_loss, kld_loss = venc.get_loss(Xs[i], y=C)
            loss = recons_loss + self.args.vspecific.kld_weight * kld_loss
            # print('recon:', recons_loss.item(), 'kld', kld_loss.item())
            tot_loss.append(loss)

        return tot_loss
    
    def get_loss(self, Xs):
        if self.args.consistency.enable:
            with torch.no_grad():
                C = self.consis_enc.consistency_features(Xs)
        else:
            C = None
        return_details = {}
        losses = []
        for i in range(self.views):
            venc = self.__getattr__(f"venc_{i+1}")
            mi_est = self.__getattr__(f"mi_est_{i+1}")
            
            # recons_loss, kld_loss = venc.get_loss(Xs[i], y=C)
            mi_lb = mi_est.learning_loss(venc.latent(Xs[i]), C)
            disent_loss = self.args.disent.lam * mi_lb
            
            # return_details[f'v{i+1}-recon-loss'] = recons_loss.item()
            # return_details[f'v{i+1}-kld-loss'] = kld_loss.item()
            return_details[f'v{i+1}-disent-loss'] = disent_loss.item()
            
            # loss = (recons_loss + kld_loss + disent_loss)
            loss = disent_loss
            return_details[f'v{i+1}-total-loss'] = loss.item()
            losses.append(loss)
            
        return losses, return_details
    

    def __fusion(self, Xs, ftype='C'):
        if self.args.consistency.enable:
            with torch.no_grad():
                consis_features = self.consis_enc.consistency_features(Xs)

        vspecific_features = []
        if self.args.vspecific.enable:
            for i in range(self.views):
                venc = self.__getattr__(f"venc_{i+1}")
                feature = venc.latent(Xs[i])
                vspecific_features.append(feature)

        if ftype == 'C':
            features = consis_features
        elif ftype == "V":
            features = vspecific_features[self.args.vspecific.best_view]
        elif ftype == "CV":
            best_view_features = vspecific_features[self.args.vspecific.best_view]
            features = torch.cat([consis_features, best_view_features], dim=-1)
        else:
            raise ValueError("Less than one kind information available.")

        return features

    def enc_dec(self, Xs):
        if self.args.consistency.enable:
            with torch.no_grad():
                C = self.consis_enc.consistency_features(Xs)
        else:
            C = None
           
        outs = []
        for i in range(self.views):
            venc = self.__getattr__(f"venc_{i+1}")
            out, _, _ = venc(Xs[i], y=C)
            outs.append(out)
        return outs

    def generate(self, z):
        outs = []
        for i in range(self.views):
            venc = self.__getattr__(f"venc_{i+1}")
            out = venc.decode(z)
            outs.append(out)
        return torch.cat(outs)

    @torch.no_grad()
    def commonZ(self, Xs):
        return self.__fusion(Xs, ftype=self.args.fusion.type)
    
    
    def all_features(self, Xs):
        batch = Xs[0].shape[0]
        if self.args.consistency.enable:
            with torch.no_grad():
                C = self.consis_enc.consistency_features(Xs)
        else:
            C = torch.zeros(batch, self.c_dim).to(self.device)
        if self.args.vspecific.enable:
            vspecific_features = []
            for i in range(self.views):
                venc = self.__getattr__(f"venc_{i+1}")
                feature = venc.latent(Xs[i])
                vspecific_features.append(feature)
            # venc = self.__getattr__(f"venc_{self.args.vspecific.best_view+1}")
            # V = venc.latent(Xs[self.args.vspecific.best_view])
            all_V = torch.cat(vspecific_features, dim=-1)
            V = vspecific_features[self.args.vspecific.best_view]
        else:
            V = torch.zeros(batch, self.v_dim).to(self.device)
            all_V = V
        return C, V, torch.cat([C, V], dim=-1), all_V

    @torch.no_grad()
    def consistency_features(self, Xs):
        return self.__fusion(Xs, ftype='C')

    @torch.no_grad()
    def vspecific_features(self, Xs, single=False):
        vspecific_features = []
        if self.args.vspecific.enable:
            for i in range(self.views):
                venc = self.__getattr__(f"venc_{i+1}")
                feature = venc.latent(Xs[i])
                vspecific_features.append(feature)
        if single:
            return vspecific_features[self.args.vspecific.best_view]
        return vspecific_features
