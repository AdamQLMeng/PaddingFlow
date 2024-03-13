dataset_list=('power' 'gas' 'hepmass' 'miniboone' 'bsds300')

for ds in "${dataset_list[@]}"; do
  # shellcheck disable=SC2045
  for e in `ls experiments/cnf/${ds}`; do
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
    if [ $ds = "power" ]; then
      python train_tabular.py \
        --data power \
        --nhidden 2 \
        --hdim_factor 20 \
        --num_blocks 1 \
        --nonlinearity softplus \
        --batch_size 1000 \
        --lr 1e-3 \
        --evaluate \
        --testset_split_len 20000 \
        --times_sample_test 100 \
        --noise_type ${noise_type} \
        --padding_dim ${padding_dim} \
        --padding_noise_scale ${padding_noise_scale} \
        --fixed_noise_scale ${fixed_noise_scale}
    fi
    if [ $ds = "gas" ]; then
      python train_tabular.py \
        --data gas \
        --nhidden 2 \
        --hdim_factor 20 \
        --num_blocks 1 \
        --nonlinearity softplus \
        --batch_size 1000 \
        --lr 1e-3 \
        --evaluate \
        --testset_split_len 20000 \
        --times_sample_test 100 \
        --noise_type ${noise_type} \
        --padding_dim ${padding_dim} \
        --padding_noise_scale ${padding_noise_scale} \
        --fixed_noise_scale ${fixed_noise_scale}
    fi
    if [ $ds = "hepmass" ]; then
      python train_tabular.py \
        --data hepmass \
        --nhidden 2 \
        --hdim_factor 10 \
        --num_blocks 10 \
        --nonlinearity softplus \
        --batch_size 10000 \
        --lr 1e-3 \
        --evaluate \
        --testset_split_len 20000 \
        --times_sample_test 100 \
        --noise_type ${noise_type} \
        --padding_dim ${padding_dim} \
        --padding_noise_scale ${padding_noise_scale} \
        --fixed_noise_scale ${fixed_noise_scale}
    fi
    if [ $ds = "miniboone" ]; then
      python train_tabular.py \
        --data miniboone \
        --nhidden 2 \
        --hdim_factor 20 \
        --num_blocks 1 \
        --nonlinearity softplus \
        --batch_size 1000 \
        --lr 1e-3 \
        --evaluate \
        --testset_split_len 20000 \
        --times_sample_test 100 \
        --noise_type ${noise_type} \
        --padding_dim ${padding_dim} \
        --padding_noise_scale ${padding_noise_scale} \
        --fixed_noise_scale ${fixed_noise_scale}
    fi
    if [ $ds = "bsds300" ]; then
      python train_tabular.py \
        --data bsds300 \
        --nhidden 2 \
        --hdim_factor 20 \
        --num_blocks 1 \
        --nonlinearity softplus \
        --batch_size 1000 \
        --lr 1e-3 \
        --evaluate \
        --testset_split_len 20000 \
        --times_sample_test 100 \
        --noise_type ${noise_type} \
        --padding_dim ${padding_dim} \
        --padding_noise_scale ${padding_noise_scale} \
        --fixed_noise_scale ${fixed_noise_scale}
    fi
  done
done