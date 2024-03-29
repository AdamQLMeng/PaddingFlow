# PaddingFLow: Improving Normalizing Flows with Padding-Dimensional Noise
 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/paddingflow-improving-normalizing-flows-with/density-estimation-on-bsds300)](https://paperswithcode.com/sota/density-estimation-on-bsds300?p=paddingflow-improving-normalizing-flows-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/paddingflow-improving-normalizing-flows-with/density-estimation-on-caltech-101)](https://paperswithcode.com/sota/density-estimation-on-caltech-101?p=paddingflow-improving-normalizing-flows-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/paddingflow-improving-normalizing-flows-with/density-estimation-on-freyfaces)](https://paperswithcode.com/sota/density-estimation-on-freyfaces?p=paddingflow-improving-normalizing-flows-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/paddingflow-improving-normalizing-flows-with/density-estimation-on-mnist)](https://paperswithcode.com/sota/density-estimation-on-mnist?p=paddingflow-improving-normalizing-flows-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/paddingflow-improving-normalizing-flows-with/density-estimation-on-omniglot)](https://paperswithcode.com/sota/density-estimation-on-omniglot?p=paddingflow-improving-normalizing-flows-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/paddingflow-improving-normalizing-flows-with/density-estimation-on-uci-gas)](https://paperswithcode.com/sota/density-estimation-on-uci-gas?p=paddingflow-improving-normalizing-flows-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/paddingflow-improving-normalizing-flows-with/density-estimation-on-uci-hepmass)](https://paperswithcode.com/sota/density-estimation-on-uci-hepmass?p=paddingflow-improving-normalizing-flows-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/paddingflow-improving-normalizing-flows-with/density-estimation-on-uci-miniboone)](https://paperswithcode.com/sota/density-estimation-on-uci-miniboone?p=paddingflow-improving-normalizing-flows-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/paddingflow-improving-normalizing-flows-with/density-estimation-on-uci-power)](https://paperswithcode.com/sota/density-estimation-on-uci-power?p=paddingflow-improving-normalizing-flows-with)

Code for reproducing the experiments in the paper:
>Qinglong Meng, Chongkun Xia, and Xueqian Wang, "PaddingFLow: Improving Normalizing Flows with Padding-Dimensional Noise
"
>
>[![arxiv.org](https://img.shields.io/badge/cs.LG-%09arXiv%3A2403.08216-red)](https://arxiv.org/abs/2403.08216)

## Setup
Setup environment:
```
bash env.sh
```
For evaluation:
```
bash install.sh
```
## Datasets

### Tabular (UCI + BSDS300)
Follow instructions from https://github.com/gpapamak/maf and place them in `tabular_vae/data/`.

### VAE datasets
Follow instructions from https://github.com/riannevdberg/sylvester-flows and place them in `tabular_vae/data/`.

### IK dataset for Panda manipulator
```
python ik/build_dataset.py --robot_name=panda --training_set_size=25000000 --only_non_self_colliding
```
## Train

Different scripts are provided for different datasets. To see all options, use the `-h` flag.

### Toy 2d:
PaddingFLow (unconditonal):
```
python tabular_vae/train_toy.py --data circles --noise_type padding
```
FFJORD (unconditonal):
```
python tabular_vae/train_toy.py --data circles --noise_type none
```
PaddingFLow (conditonal):
```
python toy_cond/train.py --data circles_conditional --noise_type padding
```
FFJORD (conditonal):
```
python toy_cond/train.py --data circles_conditional --noise_type none
```
SoftFlow (unconditonal and conditonal):
```
python toy_cond/train.py --data circles --noise_type soft
```
### Add PaddingFlow noise for tabular and vae experiments:
```
--noise_type padding --padding_dim 2 --fixed_noise_scale 0.01
```
### Tabular datasets from [MAF](https://github.com/gpapamak/maf):
power
```
python tabular_vae/train_tabular.py --data power --nhidden 2 --hdim_factor 20 --num_blocks 1 --nonlinearity softplus --batch_size 1000 --lr 1e-3
```
gas
```
python tabular_vae/train_tabular.py --data gas --nhidden 2 --hdim_factor 20 --num_blocks 1 --nonlinearity softplus --batch_size 1000 --lr 1e-3
```
hepmass
```
python tabular_vae/train_tabular.py --data hepmass --nhidden 2 --hdim_factor 10 --num_blocks 10 --nonlinearity softplus --batch_size 10000 --lr 1e-3
```
miniboone
```
python tabular_vae/train_tabular.py --data miniboone --nhidden 2 --hdim_factor 20 --num_blocks 1 --nonlinearity softplus --batch_size 1000 --lr 1e-3
```
bsds300
```
python tabular_vae/train_tabular.py --data bsds300 --nhidden 2 --hdim_factor 20 --num_blocks 1 --nonlinearity softplus --batch_size 1000 --lr 1e-3
```

### VAE Experiments:
mnist
```
python tabular_vae/train_vae_flow.py --dataset mnist --flow cnf_rank --rank 64 --dims 1024-1024 --num_blocks 2 --nonlinearity softplus
```
omniglot
```
python tabular_vae/train_vae_flow.py --dataset omniglot --flow cnf_rank --rank 20 --dims 512-512 --num_blocks 5 --nonlinearity softplus
```
freyfaces
```
python tabular_vae/train_vae_flow.py --dataset freyfaces --flow cnf_rank --rank 20 --dims 512-512 --num_blocks 2 --nonlinearity softplus
```
caltech
```
python tabular_vae/train_vae_flow.py --dataset caltech --flow cnf_rank --rank 20 --dims 2048-2048 --num_blocks 1 --nonlinearity tanh
```


### IK Experiments:
PaddingFlow
```
python ik/train.py --robot_name=panda --dim_latent_space=8 --noise_type=padding --softflow_noise_scale=0.0 --padding_scale=2
```
IKFlow (SoftFlow)
```
python ik/train.py --robot_name=panda --dim_latent_space=8 --noise_type=soft
```
Glow
```
python ik/train.py --robot_name=panda --dim_latent_space=7 --noise_type=none
```

## Evaluate
Tabular datasets:
```
bash tabular_vae/evaluate_all_tabular.sh
```
VAE Experiments:
```
bash tabular_vae/evaluate_all_vae.sh
```
IK Experiments:
```
python ik/evaluate.py --model_file ${checkpoint_file}
```



## References
- FFJORD: https://github.com/rtqichen/ffjord
- SoftFlow: https://github.com/ANLGBOY/SoftFlow
- IKFlow: https://github.com/jstmn/ikflow
- PointFlow: https://github.com/stevenygd/PointFlow
