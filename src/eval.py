import warnings
warnings.filterwarnings('ignore')
import argparse
import os
from collections import defaultdict

import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import numpy as np
import wandb
from tqdm import tqdm
from torchinfo import summary
from scipy.optimize import linear_sum_assignment
from configs.basic_config import get_cfg
from models.mrdd import MRDD
from sklearn import metrics
from utils.datatool import (get_val_transformations,
                      get_train_dataset,
                      get_val_dataset)
from utils.visualization import plot_scatter

def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def clustering_metric(y_true, y_pred, decimals=4):
    """Get clustering metric"""
    
    # ACC
    acc = clustering_accuracy(y_true, y_pred)
    acc = np.round(acc, decimals)
    
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)

    return acc, nmi, ari


def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    """Get classification metric"""
   
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)

    # precision
    precision = metrics.precision_score(y_true, y_pred, average=average)
    precision = np.round(precision, decimals)

    # F-score
    f_score = metrics.f1_score(y_true, y_pred, average=average)
    f_score = np.round(f_score, decimals)

    return accuracy, precision, f_score

@torch.no_grad()
def reconstruction(model, original):
    vspecific_recons = model(original)
    consist_recons, _ = model.consis_enc(original)
    grid = []
    for x, r in zip(original, vspecific_recons):
        grid.append(torch.cat([x, r]).detach().cpu())
    vspec_grid = make_grid(torch.cat(grid).detach().cpu())
    
    grid = []
    for x, r in zip(original, consist_recons):
        grid.append(torch.cat([x, r]).detach().cpu())
    consist_grid = make_grid(torch.cat(grid).detach().cpu())
    return consist_grid, vspec_grid


@torch.no_grad()
def extract_features(val_dataloader, model, device):
    targets = []
    consist_reprs = []
    vspecific_reprs = []
    concate_reprs = []
    all_vs = []
    for Xs, target in val_dataloader:
        Xs = [x.to(device) for x in Xs]
        consist_repr_, vspecific_repr_, concate_repr_, all_v = model.all_features(Xs)
        targets.append(target)
        consist_reprs.append(consist_repr_.detach().cpu())
        vspecific_reprs.append(vspecific_repr_.detach().cpu())
        concate_reprs.append(concate_repr_.detach().cpu())
        all_vs.append(all_v)
    targets = torch.concat(targets, dim=-1).numpy()
    consist_reprs = torch.vstack(consist_reprs).detach().cpu()
    vspecific_reprs = torch.vstack(vspecific_reprs).detach().cpu()
    concate_reprs = torch.vstack(concate_reprs).detach().cpu()
    all_vs = torch.vstack(all_vs).detach().cpu()
    return consist_reprs, vspecific_reprs, concate_reprs, all_vs, targets

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=str, help='Config File')
    args = parser.parse_args()
    return args

def main():
    # Load arguments.
    args = parse_args()
    config = get_cfg(args.config_file)
    device = torch.device(f'cuda:{config.train.devices[0]}')
    
    
    val_transformations = get_val_transformations(config)
    train_dataset = get_train_dataset(config, val_transformations)

    val_dataloader = DataLoader(train_dataset,
                                num_workers=config.train.num_workers,
                                batch_size=config.train.batch_size,
                                sampler=None,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False)
    
    run_times = 10
    n_clusters = config.dataset.class_num
    need_classification = False
    evaluation = True
    visualization = False
    pic_format = 'pdf'
    model_path = config.eval.model_path
    model = MRDD(config, consistency_encoder_path=None, device=device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # summary(model)
    
    model = model.to(device)
    print(f'Use: {device}')
    
    model.eval()
    consistency, best_specific, best_concat, all_specific, labels = extract_features(val_dataloader, model, device)
    
    z = best_concat.numpy()
    all_cat_z = torch.cat([consistency, all_specific], dim=-1)
    
    latents = {
        'consistency': consistency,
        'specificity': all_specific,
        'best_spec': best_specific,
        'labels': labels
    }
    
    torch.save(latents, os.path.join(config.train.log_dir, f'{config.dataset.name}.pkl'))
    
    if evaluation:
        print('Run consist...')
        report(run_times, n_clusters, need_classification, labels, consistency.numpy())

        print('Run best concat z...')
        report(run_times, n_clusters, need_classification, labels, z)
        
        print('Run all concat z...')
        report(run_times, n_clusters, need_classification, labels, all_cat_z)
        
        vspecs1 = []
        vspecs2 = []
        if config.views == 3:
            vspecs3 = []
        for Xs, _ in val_dataloader:
            Xs = [x.to(device) for x in Xs]
            vs = model.vspecific_features(Xs)
            vspecs1.append(vs[0].detach().cpu())
            vspecs2.append(vs[1].detach().cpu())
            if config.views == 3:
                vspecs3.append(vs[2].detach().cpu())
            
        vspecs1 = torch.vstack(vspecs1).detach().cpu()
        vspecs2 = torch.vstack(vspecs2).detach().cpu()
        if config.views == 3: 
            vspecs3 = torch.vstack(vspecs3).detach().cpu()
            
        print('Run view specificity 1')
        report(run_times, n_clusters, need_classification, labels, vspecs1.numpy())
        
        print('Run view specificity 2')
        report(run_times, n_clusters, need_classification, labels, vspecs2.numpy())
        
        if config.views == 3: 
            print('Run view specificity 3')
            report(run_times, n_clusters, need_classification, labels, vspecs3.numpy())
        
    if visualization:
        dl = DataLoader(train_dataset, 32, shuffle=True)
        recon_samples = next(iter(dl))[0]
        recon_samples = [x.to(device, non_blocking=True) for x in recon_samples]
        consist_grid, vspec_grid = reconstruction(model, recon_samples)
        save_image(consist_grid, os.path.join(config.train.log_dir, f'{config.dataset.name}-consist-recons.{pic_format}'), format=pic_format)
        save_image(vspec_grid, os.path.join(config.train.log_dir, f'{config.dataset.name}-vspec-recons.{pic_format}'), format=pic_format)
        
        
        # tsne
        tsne_samples = 2000
        
        idx = torch.rand(consistency.size(0)).argsort()
        select_idx = idx[:tsne_samples]
        
        select_consist = consistency[select_idx, :]
        select_cat = best_concat[select_idx, :]
        select_labels = labels[select_idx]
        
        from sklearn.manifold import TSNE
        from matplotlib import pyplot as plt
        
        
        _, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 4))
    
        print("Run t-sne.....")
        tsne = TSNE(n_components=2, perplexity=15, learning_rate=10)
        sz = tsne.fit_transform(select_consist.numpy())
        plot_scatter(ax1, sz, select_labels)
        handles1, ll1 = ax1.get_legend_handles_labels()
        
        bc = tsne.fit_transform(select_cat.numpy())

        plot_scatter(ax2, bc, select_labels)
  
        
        plt.figlegend(handles1, ll1, loc='lower center', bbox_to_anchor=(0.5, 0.96),fancybox=True, shadow=False, ncol=10, markerscale=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.train.log_dir, f'{config.dataset.name}-tsne.{pic_format}'), bbox_inches='tight', format=pic_format)
        plt.close()
        
        
        


def report(run_times, n_clusters, need_classification, labels, z):
    cluster_acc = []
    cluster_nmi = []
    cluster_ari = []
    cls_acc = []
    cls_p = []
    cls_fs = []
    
    for run in range(run_times):
        km = KMeans(n_clusters=n_clusters, n_init='auto')
        preds = km.fit_predict(z)
        acc, nmi, ari = clustering_metric(labels, preds)
        cluster_acc.append(acc)
        cluster_nmi.append(nmi)
        cluster_ari.append(ari)
        
        if need_classification:
            X_train, X_test, y_train, y_test = train_test_split(z, labels, test_size=0.2)
            svc = SVC()
            svc.fit(X_train, y_train)
            preds = svc.predict(X_test)
            accuracy, precision, f_score = classification_metric(y_test, preds)
            cls_acc.append(accuracy)
            cls_p.append(precision)
            cls_fs.append(f_score)
            
    print(f'[Clustering] acc: {np.mean(cluster_acc):.4f} ({np.std(cluster_acc):.4f}) | nmi: {np.mean(cluster_nmi):.4f} ({np.std(cluster_nmi):.4f}) \
    | ari: {np.mean(cluster_ari):.4f} ({np.std(cluster_ari):.4f})')

    if need_classification:
        print(f'[Classification] acc: {np.mean(cls_acc):.4f} ({np.std(cls_acc):.4f}) | fscore: {np.mean(cls_fs):.4f} ({np.std(cls_fs):.4f}) \
    | p: {np.mean(cls_p):.4f} ({np.std(cls_p):.4f}) ')
    
    return cluster_acc,cluster_nmi,cluster_ari,cls_acc,cls_p,cls_fs


    
if __name__ == '__main__':
    main()