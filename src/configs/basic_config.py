
import os

from yacs.config import CfgNode as CN


# Basic config, including basic information, dataset, and training setting.
_C = CN()

# project name, for wandb's records. **Required**
_C.project_name = 'CVPR24'
# project description, what problem does this project tackle?
_C.project_desc = 'cvpr new try.' 
# seed
_C.seed = 3407
# print log
_C.verbose = True
# enable wandb:
_C.wandb = True
# runtime
_C.runtimes = 1
# experiment name.
_C.experiment_name = "None"
# For multi-view setting
_C.views = 2
# Experiment Notes.
_C.note = ""


# Network setting.
_C.backbone = CN()
_C.backbone.type = "cnn"
# normalizations ['batch', 'layer']
_C.backbone.normalization = 'batch'
# default 'kaiming'.
_C.backbone.init_method = 'kaiming'


# For dataset
_C.dataset = CN()
# ['Scene-15', 'LandUse-21', 'Caltech101-20', 'NoisyMNIST', 
# 'EdgeMnist', 'FashionMnist', 'coil-20', 'coil-100', 'DHA23', "UWA30"]
_C.dataset.name = 'EdgeMnist'
# rootdir
_C.dataset.root = './data'
# class_num
_C.dataset.class_num = 10



# For augmentation
# training augmentation
_C.training_augmentation = CN()
_C.training_augmentation.enable = True
_C.training_augmentation.crop_size = 32
# Need training augmentation? such as mnist we suggest set as false.
_C.training_augmentation.hflip = True
# random resize crop:
_C.training_augmentation.random_resized_crop = CN()
_C.training_augmentation.random_resized_crop.size = 32
_C.training_augmentation.random_resized_crop.scale = [0.2, 1.0]
# color jitter random apply:
_C.training_augmentation.color_jitter_random_apply = CN()
_C.training_augmentation.color_jitter_random_apply.p = 0.8
# color jitter
_C.training_augmentation.color_jitter = CN()
_C.training_augmentation.color_jitter.brightness = 0.4
_C.training_augmentation.color_jitter.contrast = 0.4
_C.training_augmentation.color_jitter.saturation = 0.4
_C.training_augmentation.color_jitter.hue = 0.1
# random_grayscale
_C.training_augmentation.random_grayscale = CN()
_C.training_augmentation.random_grayscale.p = 0.2


# validation augmentation
_C.valid_augmentation = CN()
# center crop size
_C.valid_augmentation.crop_size = 32



# for training.
_C.train = CN()
_C.train.epochs = 100
_C.train.batch_size = 512
_C.train.optim = 'sgd'
_C.train.devices = [0, 1]
_C.train.lr = 0.001
_C.train.num_workers = 2
_C.train.save_log = True
# if None, it will be set as './experiments/results/[model name]/[dataset name]'
_C.train.log_dir = ""
# the interval of evaluate epoch, defaults to 5.
_C.train.evaluate = 5
# Learning rate scheduler, [cosine, step]
_C.train.scheduler = 'cosine'
_C.train.lr_decay_rate = 0.1
_C.train.lr_decay_epochs = 30
# samling num.
_C.train.samples_num = 6
# using checkpoint training.
_C.train.resume = False
_C.train.ckpt_path = ""
_C.train.use_ddp = True
_C.train.masked_ratio = 0.6
_C.train.mask_patch_size = 2

# commom feature pooling method. mean, sum, or first
_C.fusion = CN()
# C -> only consistency, V -> only view-specific, CV -> [C x V]
_C.fusion.type = 'CV'
_C.fusion.pooling_method = 'sum'
# view specific fusion weight
_C.fusion.vs_weights = 1.


# for consistency encoder setting.
_C.consistency = CN()
# 
_C.consistency.enable = True
# 
_C.consistency.continous = True
# consistency bottleneck dim.
_C.consistency.c_dim = 64
# 
_C.consistency.in_channel = 1
# 
_C.consistency.ch_mult = [1, 2, 4, 8]
# 
_C.consistency.block_size = 8
#
_C.consistency.basic_hidden_dim = 32
#
_C.consistency.latent_ch = 10
#
_C.consistency.num_res_blocks = 3
#
_C.consistency.temperature = 0.5
#
_C.consistency.kld_weight = 0.00025
# for categories vae
_C.consistency.alpha = 1.0
#
_C.consistency.consistency_encoder_path = 'path'


# for view-specific encoder setting.
_C.vspecific = CN()
# 
_C.vspecific.enable = True
# Basic hidden dim
_C.vspecific.basic_hidden_dim = 32
# Each layer output channel, will multiply with basic hidden dim.
_C.vspecific.ch_mult = [1, 2, 4, 8]
# Image shape // 2 * len(ch_mult), for example, if the image shape is 64x64, and the 
# len(ch_mult) is 8, then we have the block_size = 64/2*4 = 8.
_C.vspecific.block_size = 8
# encoder output channel.
_C.vspecific.latent_ch = 10
# z latent dim
_C.vspecific.v_dim = 100
_C.vspecific.kld_weight = 0.00025
# best view for concatenating consistency representation and view-specific representation.
_C.vspecific.best_view = 0
# num_res_blocks for the number of the Encoder and Decoder's residual block.
_C.vspecific.num_res_blocks = 2



# disentanglement
_C.disent = CN()
_C.disent.lam = 1.
_C.disent.hidden_size = 400
# for consistency dim.
_C.disent.alpha = 0.01
_C.disent.mode = 'bias'

# evaluation.
_C.eval = CN()
_C.eval.model_path = ''



def get_cfg(config_file_path):
    """
    Initialize configuration.
    """
    config = _C.clone()
    # merge specific config.
    config.merge_from_file(config_file_path)
    
    if not config.train.log_dir:
        path = f'./experiments/{config.experiment_name}'
        os.makedirs(path, exist_ok=True)
        config.train.log_dir = path
    else:
        os.makedirs(config.train.log_dir, exist_ok=True)
    # config.freeze()
    return config