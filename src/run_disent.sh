#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1


JOB_NAME='emnist'
GPUS=1

# torchrun --nproc_per_node=$GPUS --master-port=23456 train_disentangle.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/EdgeMNIST/disent.yaml'
# torchrun --nproc_per_node=$GPUS --master-port=23456 train_disentangle.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/FMNIST/disent.yaml'
# torchrun --nproc_per_node=$GPUS --master-port=23423 train_disentangle.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/coil-20/disent.yaml'
# torchrun --nproc_per_node=$GPUS --master-port=23424 train_disentangle.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/coil-100/disent.yaml'
# torchrun --nproc_per_node=$GPUS --master-port=23424 train_disentangle.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/office-31/disent.yaml'


# Single device


# python train_disentangle.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/EdgeMNIST/disent.yaml'
# python train_disentangle.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/FMNIST/disent.yaml'
# python train_disentangle.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/coil-20/disent.yaml'
# python train_disentangle.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/coil-100/disent.yaml'
python train_disentangle.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/office-31/disent.yaml'
