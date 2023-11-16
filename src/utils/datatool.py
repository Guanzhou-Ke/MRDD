
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import os

import cv2
# provent the depandency of multiple threads.
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


# -------------------------------------------------------------------------------
#                      Functional area.
# -------------------------------------------------------------------------------


def plain_transforms(img):
    """
    plain transformation operation.
    """
    return img


def coil(root, n_objs=20, n_views=3):
    """
    Download: 
    https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php

    1. coil-20: 
    http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-unproc.zip


    2. coil-100:
    http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip
    """
    if os.path.isfile(os.path.join(root, f"coil-{n_objs}/{n_views}v-cache.pth")):
        print('load cache')
        X_train, X_test, y_train, y_test = torch.load(os.path.join(root, f"coil-{n_objs}/{n_views}v-cache.pth"))
        return X_train, X_test, y_train, y_test
    else:
        from skimage.io import imread
        from sklearn.model_selection import train_test_split
        assert n_objs in [20, 100]
        data_dir = os.path.join(root, f"coil-{n_objs}")
        img_size = (1, 128, 128) if n_objs == 20 else (3, 128, 128)
        n_imgs = 72

        n = (n_objs * n_imgs) // n_views

        views = []
        labels = []

        img_idx = np.arange(n_imgs)

        for obj in range(n_objs):
            obj_list = []
            obj_img_idx = np.random.permutation(
                img_idx).reshape(n_views, n_imgs // n_views)
            labels += (n_imgs // n_views) * [obj]

            for view, indices in enumerate(obj_img_idx):
                sub_view = []
                for i, idx in enumerate(indices):
                    if n_objs == 20:
                        fname = os.path.join(data_dir, f"obj{obj + 1}__{idx}.png")
                        img = imread(fname)[None, ...]
                    else:
                        fname = os.path.join(
                            data_dir, f"obj{obj + 1}__{idx * 5}.png")
                        img = imread(fname)
                    if n_objs == 100:
                        img = np.transpose(img, (2, 0, 1))
                    sub_view.append(img)
                obj_list.append(np.array(sub_view))
            views.append(np.array(obj_list))
        views = np.array(views)
        views = np.transpose(views, (1, 0, 2, 3, 4, 5)
                            ).reshape(n_views, n, *img_size)
        cp = views.reshape(-1, *img_size)
        # print(cp.shape)
        # print(cp[:, 0, :].mean(), cp[:, 0, :].std())
        labels = np.array(labels)
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(
            list(range(n)), labels, test_size=0.2, random_state=42)
        X_train, X_test = views[:, X_train_idx, :, :, :], views[:, X_test_idx, :, :, :] 
        # torch.save((X_train, X_test, y_train, y_test), os.path.join(root, f"coil-{n_objs}/{n_views}v-cache.pth"))
        # print('save cahce.')
        return X_train, X_test, y_train, y_test


def get_train_transformations(args, task='pretext'):
    need_hflip = args['training_augmentation'].hflip
    if task == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(
                **args['training_augmentation']['random_resized_crop']),
            transforms.RandomHorizontalFlip() if need_hflip else plain_transforms,
            transforms.ToTensor()
        ])
    elif task == 'pretext':
        # Augmentation strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomResizedCrop(
                **args['training_augmentation']['random_resized_crop']),
            transforms.RandomHorizontalFlip() if need_hflip else plain_transforms,
            transforms.RandomApply([
                transforms.ColorJitter(
                    **args['training_augmentation']['color_jitter'])
            ], p=args['training_augmentation']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(
                **args['training_augmentation']['random_grayscale']),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f'Invalid augmentation strategy {task}')


def get_val_transformations(args):
    return transforms.Compose([
        transforms.Resize((args.valid_augmentation.crop_size, args.valid_augmentation.crop_size)),
        transforms.ToTensor()])


def edge_transformation(img):
    """
    edge preprocess functuin.
    """
    trans = transforms.Compose([image_edge, transforms.ToPILImage()])
    return trans(img)


def image_edge(img):
    """
    :param img:
    :return:
    """
    img = np.array(img)
    dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    edge = dilation - img
    return edge


def generate_tiny_dataset(name, dataset, sample_num=100):
    """
    Tiny data set for T-SNE to visualize the representation's structure.
    Only support EdgeMNIST, FashionMNIST.
    """
    assert name in ['EdgeMnist', 'FashionMnist']
    y = dataset.targets.unique()
    x1s = []
    x2s = []
    ys = []

    for _ in y:
        idx = dataset.targets == _
        x1, x2, yy = dataset.data[0][idx,
                                     :], dataset.data[1][idx, :], dataset.targets[idx]
        x1, x2, yy = x1[:sample_num], x2[:sample_num], yy[:sample_num]
        x1s.append(x1)
        x2s.append(x2)
        ys.append(yy)

    x1s = torch.vstack(x1s)
    x2s = torch.vstack(x2s)
    ys = torch.concat(ys)

    tiny_dataset = {
        "x1": x1s,
        "x2": x2s,
        "y": ys
    }
    os.makedirs("./experiments/tiny-data/", exist_ok=True)
    torch.save(tiny_dataset, f'./experiments/tiny-data/{name}_tiny.plk')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def align_office31(root):
    """
    This function will auto fill the lack data by color jitter method.
    The original Office dataset has 4,110 items. After alignment, it has 8,451 items
    """
    from tqdm import tqdm
    from glob import glob
    from torchvision.utils import save_image
    from sklearn.model_selection import train_test_split
    import json
    
    def padding_images(transform, image_dir, max_num):
        if len(image_dir) == max_num:
            return None
        index = torch.arange(len(image_dir))
        repeat_time = (max_num // len(image_dir))+1
        index = index.repeat(repeat_time)

        for n, idx in enumerate(index[:max_num-len(image_dir)]):
            img_path = image_dir[idx]
            new_path = f"{img_path[:-4]}_jitter_{n}.jpg"
            image = pil_loader(img_path)
            t_image = transform(image)
            save_image(t_image, new_path)

    views_mapping = {
        'A': 'amazon/images',
        'D': 'dslr/images',
        'W': 'webcam/images'
    }

    classes = ['paper_notebook', 'desktop_computer', 'punchers', 'desk_lamp', 'tape_dispenser',
               'projector', 'calculator', 'file_cabinet', 'back_pack', 'stapler', 'ring_binder',
               'trash_can', 'printer', 'bike', 'mug', 'scissors', 'bike_helmet', 'mouse', 'bookcase',
               'pen', 'bottle', 'keyboard', 'phone', 'ruler', 'headphones', 'speaker', 'letter_tray',
               'monitor', 'mobile_phone', 'desk_chair', 'laptop_computer']

    jitter = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ColorJitter(brightness=.5, hue=.3,
                                   contrast=.3, saturation=.3),
            transforms.ToTensor()
        ]
    )
    
    for c in tqdm(classes):
        item_A = glob(f"{root}/{views_mapping['A']}/{c}/*.jpg")
        item_D = glob(f"{root}/{views_mapping['D']}/{c}/*.jpg")
        item_W = glob(f"{root}/{views_mapping['W']}/{c}/*.jpg")
        max_num = max(len(item_A), len(item_D), len(item_W))

        padding_images(jitter, item_A, max_num)
        padding_images(jitter, item_D, max_num)
        padding_images(jitter, item_W, max_num)
    
    A_path = []
    D_path = []
    W_path = []
    targets = []
    # split into train set and test set.    
    for idx, c in enumerate(classes):
        item_A = glob(f"{root}/{views_mapping['A']}/{c}/*.jpg")
        item_D = glob(f"{root}/{views_mapping['D']}/{c}/*.jpg")
        item_W = glob(f"{root}/{views_mapping['W']}/{c}/*.jpg")
        
        A_path += [p[len(root):] for p in item_A]
        D_path += [p[len(root):] for p in item_D]
        W_path += [p[len(root):] for p in item_W]
        targets += ([idx] * len(item_A))
        
    X = np.c_[[A_path, D_path, W_path]].T
    Y = np.array(targets)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    train = []
    for (a, d, w), y in zip(X_train, y_train):
        train.append((a, d, w, int(y)))
        
    test = []
    for (a, d, w), y in zip(X_test, y_test):
        test.append((a, d, w, int(y)))
        
    json.dump(train, open(f'{root}/train.json', 'w'))
    json.dump(test, open(f'{root}/test.json', 'w'))
    
    
def generate_mvc_dataset(root, views):
    from collections import Counter
    from sklearn.model_selection import train_test_split
    idx_pth = os.path.join(root, 'idx.pth')
    idx = torch.load(idx_pth)

    counter = Counter(idx)
    X = []
    Y = []
    for k, v in counter.items():
        if v >= views:
            fid = k[0]
            target = k[1]
            files = []
            for i in range(v):
                if os.path.isfile(os.path.join(root, 'images', target, f'{fid}_{i}.jpg')):
                    files.append(os.path.join('images', target, f'{fid}_{i}.jpg'))
            if len(files) >= views:
                X.append(files[:views])
                Y.append(target)
    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    obj = {'train': (X_train, y_train), 'test': (X_test, y_test)}
    torch.save(obj, os.path.join(root, f'{views}V-indices.pth'))
    print(X_train.shape, X_test.shape)
        

def generate_deepfake_dataset(root):
    import os
    from sklearn.model_selection import train_test_split
    sub_folders = ['original_sequences/youtube','manipulated_sequences/Deepfakes', 'manipulated_sequences/Face2Face', 
                    'manipulated_sequences/FaceSwap', 'manipulated_sequences/NeuralTextures']

    views = 2
    
    total = []
    targets = []
    for target, sf in enumerate(sub_folders):
        files = os.listdir(os.path.join(root, sf, 'c23', 'frames'))
        for l in files:
            pngs = os.listdir(os.path.join(root, sf, 'c23', 'frames', l))
            samples = len(pngs) // views
            idx = [[v+(s*views) for v in range(views)] for s in range(samples)]
            for item in idx:
                total.append([os.path.join(sf, 'c23', 'frames', l, pngs[i]) for i in item])
                targets.append(target)

    total = np.array(total)
    targets = np.array(targets)
    print(total.shape, targets.shape)
    X_train, X_test, y_train, y_test = train_test_split(total, targets, test_size=0.2, random_state=42)
    print(f"train set: {X_train.shape}, target: {y_train.shape}")
    print(f"test set: {X_test.shape}, target: {y_test.shape}")
    torch.save({'data': X_train, 'targets': y_train}, f"./data/ffdataset/view-{views}-train.idx")
    torch.save({'data': X_test, 'targets': y_test}, f"./data/ffdataset/view-{views}-test.idx")   

# -------------------------------------------------------------------------------
#                      Dataset Area.
# -------------------------------------------------------------------------------


class EdgeMNISTDataset(torchvision.datasets.MNIST):
    """
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 views=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):

        img = self.data[idx]
        img = Image.fromarray(img.numpy(), mode='L')

        # original-view transforms
        view0 = img
        # edge-view transforms
        view1 = edge_transformation(img)
        if self.transform:
            view0 = self.transform(view0)
            view1 = self.transform(view1)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return [view0, view1], self.targets[idx]


class EdgeFMNISTDataset(torchvision.datasets.FashionMNIST):

    def __init__(self, root: str, train: bool = True,
                 transform=None,
                 target_transform=None, download: bool = False, views=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img.numpy(), mode='L')

        # original-view transforms
        view0 = img
        # edge-view transforms
        view1 = edge_transformation(img)
        if self.transform:
            view0 = self.transform(view0)
            view1 = self.transform(view1)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return [view0, view1], self.targets[idx]


class COIL20Dataset(Dataset):

    def __init__(self, root: str, train: bool = True,
                 transform=None,
                 target_transform=None, download: bool = False, views=2) -> None:

        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(range(20))
        self.train = train
        self.views = views
        self.to_pil = transforms.ToPILImage()
        X_train, X_test, y_train, y_test = coil(
            self.root, n_objs=20, n_views=self.views)
        if self.train:
            self.data = X_train
            self.targets = torch.from_numpy(y_train).long()
        else:
            self.data = X_test
            self.targets = torch.from_numpy(y_test).long()

    def __getitem__(self, index):
        views = [np.transpose(self.data[view, index, :], (1, 2, 0))
                 for view in range(self.views)]
        target = self.targets[index]

        views = [self.to_pil(v) for v in views]

        if self.transform:
            views = [self.transform(x) for x in views]

        if self.target_transform:
            target = self.target_transform(target)

        return views, target

    def __len__(self) -> int:
        return self.data.shape[1]
    
    
class COIL100Dataset(Dataset):
    
    def __init__(self, root: str, train: bool = True,
                 transform=None,
                 target_transform=None, download: bool = False, views=2) -> None:

        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(range(100))
        self.train = train
        self.views = views
        self.to_pil = transforms.ToPILImage()
        X_train, X_test, y_train, y_test = coil(
            self.root, n_objs=100, n_views=self.views)
        if self.train:
            self.data = X_train
            self.targets = torch.from_numpy(y_train).long()
        else:
            self.data = X_test
            self.targets = torch.from_numpy(y_test).long()

    def __getitem__(self, index):
        views = [np.transpose(self.data[view, index, :], (1, 2, 0))
                 for view in range(self.views)]
        target = self.targets[index]

        views = [self.to_pil(v) for v in views]

        if self.transform:
            views = [self.transform(x) for x in views]

        if self.target_transform:
            target = self.target_transform(target)

        return views, target

    def __len__(self) -> int:
        return self.data.shape[1]


class MultiViewClothingDataset(Dataset):
    """
    **Note: Before using this dataset, you have to run the `generate_mvc_dataset` function.**
    
    Refers to: Kuan-Hsien Liu, Ting-Yen Chen, and Chu-Song Chen. 
    MVC: A Dataset for View-Invariant Clothing Retrieval and Attribute Prediction, ACM ICMR 2016.
    Total: 161260 images. 10 classes. 
    (In fact, I found that it has many fails when I downloaded them. So, subject to the actual number)
    The following the size of number is my actual number:
        2 views train size: 29706, test size: 7427
        3 views train size: 29104, test size: 7277
        4 views train size: 28903, test size: 7226
        5 views train size: 8080, test size: 2021
        6 views train size: 2263, test size: 566
    """

    def __init__(self, root: str = '/mnt/disk3/data/mvc-10', train: bool = True,
                 transform=None,
                 target_transform=None, download: bool = False, views=2) -> None:

        super().__init__()
        self.classes_name = {
            "Shirts & Tops": 0,
            "Coats & Outerwear": 1,
            "Pants": 2,
            "Dresses": 3,
            "Underwear & Intimates": 4,
            "Jeans": 5,
            "Sweaters": 6,
            "Swimwear": 7,
            "Sleepwear": 8,
            "Underwear": 9
        }
        self.target2class = {v: k for k, v in self.classes_name.items()}
        self.classes = [k for k, v in self.classes_name.items()]
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.views = views
        # image loader.
        self.loader = pil_loader

        self.indices = torch.load(os.path.join(
            self.root, f'{self.views}V-indices.pth'))
        self.data, self.targets = self.indices['train'] if self.train else self.indices['test']

    def __getitem__(self, index: int):
        try:
            raw_data = [self.loader(os.path.join(self.root, path))
                        for path in self.data[index]]
        except:
            print([os.path.join(self.root, path)
                  for path in self.data[index]])
            raise

        if self.transform:
            views_data = [self.transform(x) for x in raw_data]
        else:
            views_data = raw_data
        target = torch.tensor(self.classes_name[self.targets[index]]).long()
        if self.target_transform:
            target = self.target_transform(target)

        return views_data, target

    def __len__(self) -> int:
        return len(self.data)


class Office31(Dataset):
    """
    Before use our Office31, you should firstly run the `align_office31` function.
    After that, you will get the alignment dataset, train.json, and test.json files.
    Stats:
        Totoal number: (2817, 3) (2817,)
        Train set: (2253, 3) (2253, )
        Test set: (564, 3) (564, )
    """
    
    views_mapping = {
        'A': 'amazon/images',
        'D': 'dslr/images',
        'W': 'webcam/images'
    }

    classes = ['paper_notebook', 'desktop_computer', 'punchers', 'desk_lamp', 'tape_dispenser',
               'projector', 'calculator', 'file_cabinet', 'back_pack', 'stapler', 'ring_binder',
               'trash_can', 'printer', 'bike', 'mug', 'scissors', 'bike_helmet', 'mouse', 'bookcase',
               'pen', 'bottle', 'keyboard', 'phone', 'ruler', 'headphones', 'speaker', 'letter_tray',
               'monitor', 'mobile_phone', 'desk_chair', 'laptop_computer']

    def __init__(self, root='./data/Office31', train: bool = True,
                 transform=None, target_transform=None, download: bool = False, views=3) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.load_image_path(train)
        
        
    def load_image_path(self, train):
        import json
        if train:
            self.data = json.load(open(os.path.join(self.root, 'train.json')))
        else:
            self.data = json.load(open(os.path.join(self.root, 'test.json')))
    
    def __getitem__(self, index):
        a, d, w, target = self.data[index]
        view0 = pil_loader(os.path.join(self.root, a))
        view1 = pil_loader(os.path.join(self.root, d))
        view2 = pil_loader(os.path.join(self.root, w))
        target = torch.tensor(target).long()

        if self.transform:
            view0 = self.transform(view0)
            view1 = self.transform(view1)
            view2 = self.transform(view2)

        return [view0, view1, view2], target
  
            
    def __len__(self):
        return len(self.data)
    
    
class FFDataset(Dataset):
    """
    Base on FaceForensics++ dataset. Before using this dataset, 
    you have to run the function `generate_deepfake_dataset()`.
    Two views shape: X -> (79257, 2) Y -> (79257,)
    Three views shape: X -> (49549, 3) Y -> (49549,)
    """
    
    classes = ['youtube','Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    
    def __init__(self, root='/mnt/disk3/data/DeepfakeBench/FaceForensics++', train: bool = True,
                 transform=None, target_transform=None, download: bool = False, views=3) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.views = views
        self.load_file()
        
    
    def load_file(self):
        if self.train:
            idx = torch.load(f"./data/ffdataset/view-{self.views}-train.idx")
            self.data, self.targets = idx['data'], idx['targets']
        else:
            idx = torch.load(f"./data/ffdataset/view-{self.views}-test.idx")
            self.data, self.targets = idx['data'], idx['targets']
            
    def __getitem__(self, index):
        view0, view1, target = self.data[index, 0], self.data[index, 1], self.targets[index]
        if self.views == 3:
            viwe2 = self.data[index, 2]
        view0 = pil_loader(os.path.join(self.root, view0))
        view1 = pil_loader(os.path.join(self.root, view1))
        if self.views == 3:
            view2 = pil_loader(os.path.join(self.root, viwe2))
        target = torch.tensor(target).long()

        if self.transform:
            view0 = self.transform(view0)
            view1 = self.transform(view1)
            if self.views == 3:
                view2 = self.transform(view2)
            
        if self.views == 3:
            return [view0, view1, view2], target
        return [view0, view1], target
    
    
    def __len__(self):
        return len(self.targets)


__dataset_dict = {
    'EdgeMnist': EdgeMNISTDataset,
    'FashionMnist': EdgeFMNISTDataset,
    'mvc-10': MultiViewClothingDataset,
    'coil-20': COIL20Dataset,
    'coil-100': COIL100Dataset,
    'office-31': Office31,
    'ff++': FFDataset
}


def get_train_dataset(args, transform):
    data_class = __dataset_dict.get(args.dataset.name, None)
    if data_class is None:
        raise ValueError("Dataset name error.")
    train_set = data_class(root=args.dataset.root, train=True,
                            transform=transform, download=True, views=args.views)

    return train_set


def get_val_dataset(args, transform):
    data_class = __dataset_dict.get(args.dataset.name, None)
    if data_class is None:
        raise ValueError("Dataset name error.")
    val_set = data_class(root=args.dataset.root, train=False,
                         transform=transform, views=args.views)

    return val_set


if __name__ == '__main__':
    # dataset = MultiViewClothingDataset(train=False, views=3)
    dataset = COIL100Dataset(root='/home/xyzhang/guanzhouke/MyData', train=True, views=4)
    print(dataset[0])
    pass
    # generate_mvc_dataset('/mnt/disk3/data/mvc-10', views=2)
    # generate_mvc_dataset('/mnt/disk3/data/mvc-10', views=3)
    # generate_mvc_dataset('/mnt/disk3/data/mvc-10', views=4)
    # generate_mvc_dataset('/mnt/disk3/data/mvc-10', views=5)
    # generate_mvc_dataset('/mnt/disk3/data/mvc-10', views=6)
    
    
