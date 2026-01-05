import argparse
import shutil
import time
import json

import torch

from UNet.utils import *
from UNet.unet_model import *
from torch import nn
import os
from pathlib import Path
from LaserAblation.laser_ablation_transforms import LaserAblationTransform
from Transforms.LoadIXIFreeSurferPair.find_ixi_pairs_v1 import find_ixi_pairs_v1
from cnn import LaserNet3D
from utils import soft_laser_beam_3d_batched
import msvcrt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file-name', type=str, default='config.json')
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--ixi-path', type=str, default=None)
    args = parser.parse_args()

    config_json = None
    try:
        with open(args.config_file_name) as config_json_file:
            config_json = json.load(config_json_file)
    except Exception as e:
        print("Error loading config file: ", e)
        exit(-1)

    is_training = args.epochs > 0

    # Initialize
    input_shape = tuple(config_json['model']['input_shape'])
    R, A, S = input_shape
    optimal_laser_path = LaserNet3D(R, A, S)
    # optimal_laser_path.load_state_dict(torch.load("laser_checkpoint_0.pth"))

    optimal_laser_path.train()

    if is_training:
        opt = optim.Adam(optimal_laser_path.parameters(), lr=0.001)

    # Find all the files
    ixi_paths = find_ixi_pairs_v1(args.ixi_path)

    # Self-Supervised Training Loop
    # - Insert a tumor/anomaly into a normal brain and segment into tumor & brain structures
    # - Find the current best laser beam through the brain to maximize tumor ablation
    #   and minimize brain structure damage
    # - That score is our loss that we can differentiate and back propagate to maximize
    #   the NN.

    batch_size = 1

    print("Building Dataset (train) ...")
    train_dataset = monai.data.Dataset(
        data=ixi_paths,
        transform=LaserAblationTransform(args.config_file_name).train_transforms()
    )
    print(" ... done Building Dataset (train) ..., building DataLoader (train)")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                        num_workers=0,
                                        collate_fn=monai.data.pad_list_data_collate)
    print(" ... done Building DataLoader (train).")

    import napari

    if is_training:
        total_count = 0
        for epoch in range(args.epochs):
            for b, batch in enumerate(train_dataloader):
                if b == 0 and epoch == 0:
                    print(f"Successfully loaded batch 0.")

                draw = draw_volume_and_segmentation(
                    batch["image"][0][0],
                    batch["labels"][0][0],
                    axis=2
                )
                cv.imshow("predicted", draw)
                cv.waitKey(1)

                tumor = torch.where(batch["labels"][0][0] == 10, True, False)
                if torch.count_nonzero(tumor) == 0:
                    continue

                labels = batch["labels"].to(torch.float32)
                r0, a0, r1, a1 = optimal_laser_path(labels)
                s0 = torch.tensor(0, dtype=torch.float32, device='cuda')
                s1 = torch.tensor(S - 1, dtype=torch.float32, device='cuda')

                # reward function
                laser_tensor = soft_laser_beam_3d_batched(r0, a0, s0, r1, a1, s1, R, A, S, 31.0)
                reward = (laser_tensor * batch["reward"]).sum()
                loss = -reward

                opt.zero_grad()
                loss.backward()
                opt.step()

                print(f"Epoch: {epoch}, Loss: {loss.item()}    ||    "
                      f"{int(r0.item())} : {int(r1.item())}    ||    "
                      f"{int(a0.item())} : {int(a1.item())}    ||    "
                      f"{int(s0.item())} : {int(s1.item())}")


                total_count += batch_size

            torch.save(optimal_laser_path.state_dict(), "laser_checkpoint_" + str(epoch) + ".pth")


if __name__ == '__main__':
    main()
