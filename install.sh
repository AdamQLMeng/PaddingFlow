#! /bin/bash
# Compile CUDA kernel for CD/EMD loss
root=`pwd`
export CXXFLAGS="$(echo $CXXFLAGS | sed -e 's/ -std=[^ ]*//')"
cd tabular_vae/metrics/pytorch_structural_losses/
make clean
make
cd $root

