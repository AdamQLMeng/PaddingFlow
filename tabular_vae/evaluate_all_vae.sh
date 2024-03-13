dataset_list=('mnist' 'cifar10' 'freyfaces' 'caltech' 'omniglot')
dataset_list=('caltech' 'omniglot' 'mnist' 'cifar10')
for ds in "${dataset_list[@]}"; do
  # shellcheck disable=SC2045
  for e in `ls experiments/vae/${ds}`; do
    OLD_IFS="$IFS"  #保存当前shell默认的分割符，一会要恢复回去
    IFS="_"                  #将shell的分割符号改为，“”
    array=(${e})     #分割符是“，”，"hello,shell,split,test" 赋值给array 就成了数组赋值
    IFS="$OLD_IFS"  #恢复shell默认分割符配置
    noise_type=${array[0]}
    if [ ${#array[@]} -gt 2 ]; then
      padding_dim=${array[1]}
      padding_noise_scale=${array[2]}
      fixed_noise_scale=${array[3]}
   else
      padding_dim=0
      padding_noise_scale=0
      fixed_noise_scale=0
    fi
    echo $ds $e $noise_type $padding_dim $padding_noise_scale $fixed_noise_scale
    if [ $ds = "mnist" ]; then
      python train_vae_flow.py \
        --dataset mnist \
        --flow cnf_rank \
        --rank 64 \
        --dims 1024-1024 \
        --num_blocks 2 \
        --nonlinearity softplus \
        --evaluate \
        --noise_type ${noise_type} \
        --padding_dim ${padding_dim} \
        --padding_noise_scale ${padding_noise_scale} \
        --fixed_noise_scale ${fixed_noise_scale}
    fi
    if [ $ds = "cifar10" ]; then
      python train_vae_flow.py \
        --dataset cifar10 \
        --flow cnf_rank \
        --rank 64 \
        --dims 1024-1024 \
        --num_blocks 2 \
        --nonlinearity softplus \
        --evaluate \
        --noise_type ${noise_type} \
        --padding_dim ${padding_dim} \
        --padding_noise_scale ${padding_noise_scale} \
        --fixed_noise_scale ${fixed_noise_scale}
    fi
    if [ $ds = "freyfaces" ]; then
      python train_vae_flow.py \
        --dataset freyfaces \
        --flow cnf_rank \
        --rank 20 \
        --dims 512-512 \
        --num_blocks 2 \
        --nonlinearity softplus \
        --evaluate \
        --noise_type ${noise_type} \
        --padding_dim ${padding_dim} \
        --padding_noise_scale ${padding_noise_scale} \
        --fixed_noise_scale ${fixed_noise_scale}
    fi
    if [ $ds = "caltech" ]; then
      python train_vae_flow.py \
        --dataset caltech \
        --flow cnf_rank \
        --rank 20 \
        --dims 2048-2048 \
        --num_blocks 1 \
        --nonlinearity tanh \
        --evaluate \
        --noise_type ${noise_type} \
        --padding_dim ${padding_dim} \
        --padding_noise_scale ${padding_noise_scale} \
        --fixed_noise_scale ${fixed_noise_scale}
    fi
    if [ $ds = "omniglot" ]; then
      python train_vae_flow.py \
        --dataset omniglot \
        --flow cnf_rank \
        --rank 20 \
        --dims 512-512 \
        --num_blocks 5 \
        --nonlinearity softplus \
        --evaluate \
        --noise_type ${noise_type} \
        --padding_dim ${padding_dim} \
        --padding_noise_scale ${padding_noise_scale} \
        --fixed_noise_scale ${fixed_noise_scale}
    fi
  done
done
