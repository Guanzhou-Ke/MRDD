import argparse
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import wandb
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
from torchinfo import summary

from configs.basic_config import get_cfg
from models.consistency_models import ConsistencyAE
from utils import (clustering_by_representation,
                   reproducibility_setting, get_device, save_checkpoint, get_masked,
                   plot_training_loggers, get_masked_value)
from utils.datatool import (get_val_transformations,
                      get_train_dataset,
                      get_val_dataset)
from optimizer import get_optimizer, get_scheduler


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=str, help='Config File')
    args = parser.parse_args()
    return args


def init_distributed_mode():
    # set cuda device
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(
        backend='nccl' if dist.is_nccl_available() else 'gloo')


def clean_distributed():
    dist.destroy_process_group()


def smartprint(*msg):
    if LOCAL_RANK == 0 or LOCAL_RANK == -1:
        print(*msg)

@torch.no_grad()
def valid_by_kmeans(val_dataloader, model, use_ddp, device):
    targets = []
    consist_reprs = []

    for Xs, target in val_dataloader:
        Xs = [x.to(device) for x in Xs]
        if use_ddp:
            consist_repr_ = model.module.consistency_features(Xs)
        else:
            consist_repr_ = model.consistency_features(Xs)
        targets.append(target)
        consist_reprs.append(consist_repr_.detach().cpu())
        
    targets = torch.concat(targets, dim=-1).numpy()
    consist_reprs = torch.vstack(consist_reprs).detach().cpu().numpy()
   
    result = {}
    acc, nmi, ari, _, p, fscore = clustering_by_representation(consist_reprs, targets)
    result['consist-acc'] = acc
    result['consist-nmi'] = nmi
    result['consist-ari'] = ari
    result['consist-p'] = p
    result['consist-fscore'] = fscore

    return result


def train_a_epoch(args, train_dataloader, model, epoch, device, optimizer, lr):
    losses = defaultdict(list)
    if args.train.use_ddp:
        model.module.train()
        parameters = list(model.module.parameters())
    else:
        model.train()
        parameters = list(model.parameters())
    if args.verbose and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
        pbar = tqdm(total=len(train_dataloader)*args.views)
    mask_ratio, mask_patch_size = args.train.masked_ratio, args.train.mask_patch_size
    # mask_patch_size = args.train.mask_patch_size
    # try dynamic mask_value
    # mask_ratio = get_masked_value(epoch, end_epoch=args.train.epochs)
    for Xs, _ in train_dataloader:
        Xs = [x.to(device) for x in Xs]
  
        if args.train.use_ddp:
            loss, loss_parts = model.module.get_loss(Xs, epoch, mask_ratio, mask_patch_size)
        else:
            loss, loss_parts = model.get_loss(Xs, epoch, mask_ratio, mask_patch_size)

        for k, v in loss_parts.items():
            losses[k].append(v)    
            
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
    
        show_losses = {k: np.mean(v) for k, v in losses.items()}
        
        if args.verbose and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
            loss_str = ', '.join([f'{k}:{v:.4f}' for k, v in show_losses.items()])
            pbar.set_description(f"Training | epoch: {epoch}, lr: {lr:.4f}, {loss_str}")
            pbar.update()
            
    if args.verbose and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
        pbar.close()
        
    return show_losses, loss.item()

# missing_rate = [0.1, 0.3, 0.5, 0.7, 0.9]

def main():
    # Load arguments.
    args = parse_args()
    config = get_cfg(args.config_file)
    
    
    use_wandb = config.wandb
    use_ddp = config.train.use_ddp
    
    runtimes = config.runtimes
    evaluate_intervals = config.train.evaluate
    result_dir = os.path.join(config.train.log_dir, f'consist-c{config.consistency.c_dim}-m{config.train.masked_ratio}')
    os.makedirs(result_dir, exist_ok=True)
    
    if use_ddp:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
            [str(i) for i in config.train.devices])
    
    device = get_device(config, LOCAL_RANK)
    print(f'Use: {device}')
    
    if use_ddp:
        init_distributed_mode()
    
    # for record each running.
    running_loggers = {}
    for r in range(runtimes):
        seed = config.seed
        sub_logger = defaultdict(list)
        
        checkpoint_path = os.path.join(result_dir, f'checkpoint-{seed}.pth')
        finalmodel_path = os.path.join(result_dir, f'final_model-{seed}.pth')

        # For reproducibility
        reproducibility_setting(seed)

        val_transformations = get_val_transformations(config)
        train_dataset = get_train_dataset(config, val_transformations)
        
        if use_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                    num_workers=config.train.num_workers,
                                    batch_size=config.train.batch_size // WORLD_SIZE,
                                    sampler=train_sampler if use_ddp else None,
                                    shuffle=False if use_ddp else True,
                                    pin_memory=True,
                                    drop_last=True)
        # Only evaluation at the first device.
        if LOCAL_RANK == 0 or LOCAL_RANK == -1:
            val_dataset = get_val_dataset(config, val_transformations)
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=config.train.batch_size // WORLD_SIZE,
                                        num_workers=config.train.num_workers,
                                        shuffle=False,
                                        drop_last=False,
                                        pin_memory=True)
            print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))
        
            dl = DataLoader(val_dataset, 16, shuffle=True)
            recon_samples = next(iter(dl))[0]
            recon_samples = [x.to(device, non_blocking=True) for x in recon_samples]
        
        # Create contrastive model.
        model = ConsistencyAE(basic_hidden_dim=config.consistency.basic_hidden_dim,
                            c_dim=config.consistency.c_dim,
                            continous=config.consistency.continous,
                            in_channel=config.consistency.in_channel,
                            num_res_blocks=config.consistency.num_res_blocks,
                            ch_mult=config.consistency.ch_mult,
                            block_size=config.consistency.block_size,
                            temperature=config.consistency.temperature,
                            latent_ch=config.consistency.latent_ch,
                            kld_weight=config.consistency.kld_weight,
                            views=config.views,
                            categorical_dim=config.dataset.class_num
                            )
        if LOCAL_RANK == 0 or LOCAL_RANK == -1:
            summary(model)
        smartprint(f"model loaded! ")
        
        optimizer = get_optimizer(model.parameters(), config.train.lr, config.train.optim)
        scheduler = get_scheduler(config, optimizer)
        
        # Checkpoint
        if config.train.resume and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            model = model.to(device)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            smartprint(f"Load checkpoint {checkpoint_path} at epoch: {start_epoch}!")
        else:
            start_epoch = 0
            model = model.to(device)
            
        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[LOCAL_RANK],
                output_device=LOCAL_RANK,
                find_unused_parameters=True,
                broadcast_buffers=False
            )
        
        if use_wandb and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
            wandb.init(project=config.project_name, config=config,
                    name=f'{config.experiment_name}-consist-c{config.consistency.c_dim}-m{config.train.masked_ratio}-{seed}')
            wandb.watch(model, log='all', log_graph=True, log_freq=15)

        # Start scan training.
        best_loss = np.inf
        old_best_model_path = ""
        for epoch in range(start_epoch, config.train.epochs):

            lr = optimizer.param_groups[0]['lr']
            
            # Train
            if use_ddp:
                train_dataloader.sampler.set_epoch(epoch)
            losses, cur_loss = train_a_epoch(config, train_dataloader, model, epoch, device, optimizer, lr)
            
            if cur_loss <= best_loss:
                        best_loss = cur_loss
                        best_model_path = os.path.join(result_dir, f'best-{int(best_loss)}-{epoch}-{seed}.pth')
                        if old_best_model_path:
                            # save storage.
                            os.remove(old_best_model_path)
                        old_best_model_path = best_model_path
                        if config.train.use_ddp:
                            model.module.eval()
                            # Save final model
                            torch.save(model.module.state_dict(), best_model_path) 
                        else:
                            model.eval()
                            # Save final model
                            torch.save(model.state_dict(), best_model_path)
            
            if not config.verbose:
                smartprint(f"[Training {epoch}/{config.train.epochs}]", ', '.join([f'{k}:{v:.4f}' for k, v in losses.items()]))
            
            
            for k, v in losses.items():
                sub_logger[k].append(v)
                
            if scheduler is not None:
                scheduler.step()
            
            if use_wandb and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
                wandb.log(losses, step=epoch)
                    

            if LOCAL_RANK == 0 or LOCAL_RANK == -1:
                if epoch % evaluate_intervals == 0:
                    if config.train.use_ddp:
                        model.module.eval()
                    else:
                        model.eval()
                    rcons_grid = reconstruction(model, recon_samples, config.train.use_ddp)
                    
                    sample_grid = sampling(model, config.train.samples_num, device, use_ddp)
                    
                    kmeans_result = valid_by_kmeans(val_dataloader, model, use_ddp, device)
                    
                    
                    for k, v in kmeans_result.items():
                        sub_logger[k].append(v)
                        
                    print(f"[Evaluation {epoch}/{config.train.epochs}]", ', '.join([f'{k}:{v:.4f}' for k, v in kmeans_result.items()]))
                    
                    if use_wandb:
                        wandb.log({'rcons-grid': wandb.Image(rcons_grid)}, step=epoch)
                        wandb.log(kmeans_result, step=epoch)
                        wandb.log({'samples': wandb.Image(sample_grid)}, step=epoch)
                
                # Checkpoint
                # save_checkpoint(config, checkpoint_path, model, optimizer, None, epoch)
            if use_ddp:    
                dist.barrier()
        
        # update seed.        
        running_loggers[f'r{r+1}-{seed}'] = sub_logger
        # running_loggers[f'm-{mask_ratio}'] = sub_logger
                   
        if LOCAL_RANK == 0 or LOCAL_RANK == -1:
            # update seed.
            seed = torch.randint(1, 9999, (1, )).item()
            # to ensure the seed is self-consistent.
            config.seed = seed
            if config.train.use_ddp:
                model.module.eval()
                # Save final model
                torch.save(model.module.state_dict(), finalmodel_path) 
            else:
                model.eval()
                # Save final model
                torch.save(model.state_dict(), finalmodel_path)
                
                
    if LOCAL_RANK == 0 or LOCAL_RANK == -1:            
        torch.save(running_loggers, os.path.join(result_dir, 'loggers.pkl'))
        plot_training_loggers(running_loggers, os.path.join(result_dir, 'train_logger.png'))
        
    if use_ddp:
        clean_distributed()

@torch.no_grad()
def sampling(model, samples_num, device, use_ddp):
    """
    Sampling from conditional vaes
    """
    if use_ddp:
        outs = model.module.sampling(samples_num, device=device)
    else:
        outs = model.sampling(samples_num, device=device)
    sample_grid = make_grid(torch.cat(outs).detach().cpu())
    return sample_grid

@torch.no_grad()
def reconstruction(model, original, use_ddp):
    if use_ddp:
        recons, _ = model.module(original)
    else:
        recons, _ = model(original)
    grid = []
    for x, r in zip(original, recons):
        grid.append(torch.cat([x, r]).detach().cpu())
    grid = make_grid(torch.cat(grid).detach().cpu())
    return grid
    

    
if __name__ == '__main__':
    main()
    