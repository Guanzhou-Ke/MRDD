import torch
import numpy as np


def get_masked(batch_size, shapes, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data.
        Args:
          batch_size: the ba
          shapes: the shape of data.
          missing_rate: missing ratio. [0, 1]
        Returns: 
          mask: torch.ByteTensor
    """
    masks = []
    for shape in shapes:
        mask = np.r_[[np.random.choice([0, 1], size=shape, p=[1-missing_rate, missing_rate]) for _ in range(batch_size)]]
        masks.append(torch.BoolTensor(mask))
    return masks


def mask_image(img, patch_size=4, mask_ratio=0.5, return_img=True):
    """mask image like MAE.

    Args:
        img (Tensor): (B, C, H, W) images.
        patch_size (int, optional): masked patch size. Defaults to 4.
        mask_ratio (float, optional): mask ratio. Defaults to 0.5.
        return_img (bool, optional): Return masked image if ture, whether return return visable image.
    Returns:
        img (Tensor): (B, C, H, W) masked images.
    """
    b, c, h, w = img.shape
    patch_h = patch_w = patch_size
    num_patches = (h // patch_h) * (w // patch_w)

    patches = img.view(
        b, c,
        h // patch_h, patch_h, 
        w // patch_w, patch_w
    ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)
    
    num_masked = int(mask_ratio * num_patches)
    shuffle_indices = torch.rand(b, num_patches).argsort()
    mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]
    batch_ind = torch.arange(b).unsqueeze(-1)
    if return_img:
        # masked
        patches[batch_ind, mask_ind] = 0
        x_masked = patches.view(
            b, h // patch_h, w // patch_w, 
            patch_h, patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)
        
        return x_masked
    else:
        return patches[batch_ind, unmask_ind]


def save_checkpoint(config, checkpoint_path, model, optimizer, scheduler, epoch):
    checkpoint_state_dict = {
        'optimizer': optimizer.state_dict(),
        'epoch': epoch+1,
    }

    if config.train.use_ddp:
        checkpoint_state_dict['model'] = model.module.state_dict()
    else:
        checkpoint_state_dict['model'] = model.state_dict()

    if scheduler is not None:
        checkpoint_state_dict['scheduler'] = scheduler.state_dict()

    # Checkpoint
    torch.save(checkpoint_state_dict, checkpoint_path)



def reproducibility_setting(seed):
    """
    set the random seed to make sure reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

    print('Global seed:', seed)


def get_device(args, local_rank):
    if args.train.use_ddp:
        device = torch.device(
            f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(f"cuda:{args.train.devices[0]}") if torch.cuda.is_available(
        ) else torch.device('cpu')
    return device


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


def label_to_one_hot(label_idx, num_classes) -> torch.Tensor:
    return torch.nn.functional.one_hot(label_idx,
                                       num_classes=num_classes)


def one_hot_to_label(one_hot_arr: torch.Tensor) -> torch.Tensor:
    return one_hot_arr.argmax(dim=1)


def get_masked_value(current_epoch, start_epoch=0, end_epoch=100, start_value=0.1, end_value=0.9):
    ratio = (current_epoch - start_epoch) / (end_epoch - start_epoch) # 当前步数在总步数的比例
    # 总计100步，当前current步
    ratio = max(0, min(1, ratio))
    value = (ratio * (end_value - start_value)) + start_value
    return value
