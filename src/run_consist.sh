#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1


JOB_NAME='emnist'
GPUS=1

# torchrun --nproc_per_node=$GPUS --master-port=23456 train_consistency.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/EdgeMNIST/consist.yaml'
# torchrun --nproc_per_node=$GPUS --master-port=23456 train_consistency.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/FMNIST/consist.yaml'
# torchrun --nproc_per_node=$GPUS --master-port=23457 train_consistency.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/coil-20/consist.yaml'
# torchrun --nproc_per_node=$GPUS --master-port=23458 train_consistency.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/coil-100/consist.yaml'
# torchrun --nproc_per_node=$GPUS --master-port=23459 train_consistency.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/office-31/consist.yaml'

# Single device

# python train_consistency.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/EdgeMNIST/consist.yaml'
# python train_consistency.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/FMNIST/consist.yaml'
# python train_consistency.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/coil-20/consist.yaml'
python train_consistency.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/coil-100/consist.yaml'
# python train_consistency.py  -f '/data/home/scv9554/run/guanzhouke/my_experiments/cvpr24/src/configs/office-31/consist.yaml'

