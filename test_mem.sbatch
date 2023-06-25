#!/usr/bin/bash
#SBATCH --job-name="clip-fusion"
#SBATCH --output=%j.out
#SBATCH --time=20:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=24G
#SBATCH --tmp=32G
#SBATCH --gpus=rtx_3090:1

module load gcc/8.2.0 python_gpu/3.10.4 cuda/11.8.0 git-lfs/2.3.0 git/2.31.1 eth_proxy

SCENE_ID=scene0000_00
DATA_CFG=./examples/dataconfigs/scannet/${SCENE_ID}.yaml
SCANNET_ROOT=/cluster/project/cvg/weders/data/scannet

${SCRATCH}/.python_venv/sp3/bin/python test_mem.py \
    --dataconfig_path $DATA_CFG \
    --dataset_dir $SCANNET_ROOT \
    --sub_folders scans/${SCENE_ID}/data \
    --fusion_device cuda:0 \
    --feature_height 240 \
    --feature_width 320 
