#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=yx3017 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/yx3017/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm\
ls /homes/yx3017
cd /homes/yx3017/Desktop/Individual_project/Individual_project/MNIST\ experiements
python3 MNIST_experiement.py