import numpy as np
import torch
from torchvision.utils import make_grid


def plot_confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)
    
    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)
    
    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' %(100*z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    
def plot_scatter(ax, x, labels=None, marker='1', legend=None):
    s = 16
    if labels is not None:
        y = np.unique(labels)
        for i in y:
            idx = labels == i
            ax.scatter(x[idx, 0], x[idx, 1], s=s, marker=marker, label=f"{i}")
    else:
        ax.scatter(x[:, 0], x[:, 1], s=s, marker=marker, label=legend)
        
def plot_embed_images(ax, X, images, targets):
    from matplotlib import offsetbox
    from matplotlib import pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler().fit_transform(X)
    
    # plot_scatter(ax, X, legend="")
    
    target_names = np.unique(targets)
    for digit in target_names:
        ax.scatter(
            *X[targets == digit].T,
            marker='1',
            s=60,
            color=plt.cm.Dark2(digit),
            alpha=0.2,
            zorder=1,
        )
    
    shown_images = np.array([[1.0, 1.0]])  # just something big
    for i in range(X.shape[0]):
        # plot every digit on the embedding
        # show an annotation box for a group of digits
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r, zoom=0.3), X[i], pad=0.1,
        )
        imagebox.set(zorder=2)
        ax.add_artist(imagebox)
        
    # ax.axis("off")
        

def plot_embedding_by_tsne(valid_dataloader, model: torch.nn.Module, output_file=None):
    from matplotlib import pyplot as plt
    from sklearn.manifold import TSNE
    from tqdm import tqdm
    
    model.eval()
    consistency_reprs = []
    view1_reprs = []
    view1 = []
    view2 = []
    view2_reprs = []
    targets = []
    with torch.no_grad():
        for Xs, target in tqdm(valid_dataloader, desc='Plot TSNE'):
            view1.append(Xs[0])
            view2.append(Xs[1])
            Xs = [x.cuda(non_blocking=True) for x in Xs]
            consistency_reprs.append(model.consistency_features(Xs))
            vs = model.vspecific_features(Xs)
            view1_reprs.append(vs[0])
            view2_reprs.append(vs[1])
            targets.append(target)
    targets = torch.concat(targets, dim=-1).numpy()
    consistency_reprs = torch.vstack(consistency_reprs).squeeze().detach().cpu().numpy()
    view1 = torch.vstack(view1).squeeze().detach().cpu().numpy()
    view2 = torch.vstack(view2).squeeze().detach().cpu().numpy()
    view1_reprs = torch.vstack(view1_reprs).squeeze().detach().cpu().numpy()
    view2_reprs = torch.vstack(view2_reprs).squeeze().detach().cpu().numpy()
    _, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(15, 4))
    
    print("Run t-sne.....")
    tsne = TSNE(n_components=2, perplexity=15, learning_rate=10)
    sz = tsne.fit_transform(consistency_reprs)
    # plot_embed_images(ax1, sz, view1, targets)
    plot_scatter(ax1, sz, targets)
    ax1.legend(markerscale=2)
    
    sh1, sh2 = tsne.fit_transform(view1_reprs), tsne.fit_transform(view2_reprs)
    plot_embed_images(ax2, sh1, view1, targets)
    plot_embed_images(ax3, sh2, view2, targets)
    # plot_scatter(ax2, sh1, legend='view 1')
    # plot_scatter(ax2, sh2, legend='view 2')
    # ax2.legend(markerscale=2)
    
    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    
    
def conditional_sampling(model, cidx, sample_nums=8, views=2):
    outs = model.sampling(sample_nums, cidx)
    sample_grid = make_grid(outs.detach().cpu(), nrow=sample_nums*views)
    return sample_grid


def sampling_by_z(model, y, z, nrow=8, device='cpu'):
    z = torch.cat([z, y], dim=1)
    z = z.to(device)
    outs = model.generate(z)   
    sample_grid = make_grid(outs.detach().cpu(), nrow=nrow if nrow >= 8 else 8)
    return sample_grid


def plot_training_loggers(loggers: dict, save_path: str):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    for sub_k, sub_v in loggers.items():
        for k, v in sub_v.items():
            if 'acc' in k:
                ax.plot(v, label=f'{sub_k}-{k}')
    ax.legend()
    plt.show()
    plt.savefig(save_path)
    
    
if __name__ == '__main__':
    pass
    # import  torch
    # loggers = torch.load('/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/experiments/fmnist/consist-cont-c10-m0.1/loggers.pkl')
    # plot_training_loggers(loggers, './logger.png')
