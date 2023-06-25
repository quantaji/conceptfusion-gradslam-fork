import os
import pickle as pkl
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import torch
import torch.nn.functional as f
import tyro
from tqdm import trange

from gradslam.datasets import AzureKinectDataset, ICLDataset, ReplicaDataset, ScannetDataset, load_dataset_config
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.rgbdimages import RGBDImages


@dataclass
class ProgramArgs:

    mode: str = ""  # 'E' for feature extraction, 'F' for fusion, 'D' for deleting temp files

    # dataset
    dataconfig_path: str = (Path.cwd() / 'examples/dataconfigs/scannet/scene0000_00.yaml')
    dataset_dir: Union[str, Path] = (Path.home() / 'Datasets/ScanNet')
    sub_folders: str = 'scans/scene0000_00/data'
    odomfile: str = ''  # # Odometry file (in format created by 'realtime/compute_and_save_o3d_odom.py'), only used in azurekinect dataset, 'data/azurekinect/poses.txt'

    # fusion algorithm related
    fusion_device: str = 'cuda'  # device used for feature fusion
    start: int = 0
    end: int = -1
    stride: int = 20

    feature_height: int = 240
    feature_width: int = 320


def get_dataset(dataconfig_path, dataset_dir, sub_folders, **kwargs):
    config_dict = load_dataset_config(dataconfig_path)
    if config_dict['dataset_name'].lower() in ['icl']:
        return ICLDataset(config_dict, dataset_dir, sub_folders, **kwargs)
    elif config_dict['dataset_name'].lower() in ['replica']:
        return ReplicaDataset(config_dict, dataset_dir, sub_folders, **kwargs)
    elif config_dict['dataset_name'].lower() in ['azure', 'azurekinect']:
        return AzureKinectDataset(config_dict, dataset_dir, sub_folders, **kwargs)
    elif config_dict['dataset_name'].lower() in ['scannet']:
        return ScannetDataset(config_dict, dataset_dir, sub_folders, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


@torch.no_grad()
def gradslam_fusion(args: ProgramArgs):
    # Get dataset
    dataset = get_dataset(
        dataconfig_path=args.dataconfig_path,
        dataset_dir=args.dataset_dir,
        sub_folders=args.sub_folders,
        desired_height=args.feature_height,
        desired_width=args.feature_width,
        start=args.start,
        end=args.end,
        stride=args.stride,
        load_embeddings=False,  # We will not read in embeddings; we will compute them
        odomfile=args.odomfile,
        device=args.fusion_device,
    )
    # setting up slam
    slam = PointFusion(odom="gt", dsratio=1, device=args.fusion_device, use_embeddings=True)
    frame_cur, frame_prev = None, None
    pointclouds = Pointclouds(device=args.fusion_device)

    print("Running PointFusion (incremental mode)...")
    for idx in trange(len(dataset)):

        torch.cuda.empty_cache()
        color, depth, intrinsics, pose, *_ = dataset[idx]

        dense_feats = f.normalize(torch.randn(size=(args.feature_height, args.feature_width, 512), device=args.fusion_device), dim=-1)  # just fusing random features
        frame_cur = RGBDImages(
            color.unsqueeze(0).unsqueeze(0),
            depth.unsqueeze(0).unsqueeze(0),
            intrinsics.unsqueeze(0).unsqueeze(0),
            pose.unsqueeze(0).unsqueeze(0),
            embeddings=dense_feats.unsqueeze(0).unsqueeze(0),
        )
        pointclouds, _ = slam.step(pointclouds, frame_cur, frame_prev, inplace=True)


if __name__ == "__main__":
    args = tyro.cli(ProgramArgs)

    gradslam_fusion(args)
