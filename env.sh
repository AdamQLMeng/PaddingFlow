#! /bin/bash
# Create virtual environment
conda create -n flows python=3.9
conda activate flows
# Install dependencies
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install pandas pytorch_lightning matplotlib tqdm scikit-learn scipy tensorboardX torchdiffeq ninja imageio einops jrl==0.0.9 qpth -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install package/FrEIA-debug/
pip install PyQT5
sudo apt-get install libxcb-xinerama0
conda install theano
