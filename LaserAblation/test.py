import argparse
import shutil
import time
import json
import napari

import torch

from UNet.utils import *
from UNet.unet_model import *
from torch import nn
import os
from pathlib import Path
from LaserAblation.laser_ablation_transforms import LaserAblationTransform
from Transforms.LoadIXIFreeSurferPair.find_ixi_pairs_v1 import find_ixi_pairs_v1
import msvcrt
from cnn import LaserNet3D
from utils import soft_laser_beam_3d_batched, keep_largest_components_3d


def main():
    tumor_label = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file-name', type=str, default='config.json')
    parser.add_argument('--ixi-path', type=str, default=None)
    args = parser.parse_args()

    config_json = None
    try:
        with open(args.config_file_name) as config_json_file:
            config_json = json.load(config_json_file)
    except Exception as e:
        print("Error loading config file: ", e)
        exit(-1)

    # Initialize
    input_shape = tuple(config_json['model']['input_shape'])
    R, A, S = input_shape

    optimal_laser_path = LaserNet3D(R, A, S)
    optimal_laser_path.load_state_dict(torch.load("laser_checkpoint_1.pth"))
    optimal_laser_path.eval()

    # Find all the files
    ixi_paths = find_ixi_pairs_v1(args.ixi_path)

    batch_size = 1
    print("Building CacheDataset (test) ...")  # , cache_num=self._cache_size)
    dataset = monai.data.Dataset(
        data=ixi_paths,
        transform=LaserAblationTransform(args.config_file_name).train_transforms()
    )
    print(" ... done Building CacheDataset (test) ..., building dataLoader (train)")
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=0,
                            collate_fn=monai.data.pad_list_data_collate)
    print(" ... done Building DataLoader (train).")

    for b, batch in enumerate(dataloader):
        if b == 0:
            print(f"Successfully loaded batch 0.")

        draw = draw_volume_and_segmentation(
            batch["image"][0][0],
            batch["labels"][0][0],
            axis=2
        )
        cv.imshow("predicted", draw)
        cv.waitKey(1)

        tumor = torch.where(batch["labels"][0][0] == tumor_label, True, False)
        if torch.count_nonzero(tumor) == 0:
            continue

        r0, a0, r1, a1 = optimal_laser_path(batch["image"])
        s0 = torch.tensor(0, dtype=torch.float32, device='cuda')
        s1 = torch.tensor(S - 1, dtype=torch.float32, device='cuda')

        laser_tensor = soft_laser_beam_3d_batched(r0, a0, s0, r1, a1, s1, R, A, S, 11.0)

        print(f"{int(r0.item())} : {int(r1.item())}    ||    "
              f"{int(a0.item())} : {int(a1.item())}    ||    "
              f"{int(s0.item())} : {int(s1.item())}")

        tumor = keep_largest_components_3d(
            batch["labels"][0][0].detach().cpu().numpy().astype(np.uint32), tumor_label
        )

        NAPARI = True
        if NAPARI:  # and (total_count == 0 or total_count > 100 and b % 10 == 0):

            viewer = napari.Viewer()

            # Image.
            viewer.add_image(
                batch["image"][0].detach().cpu().numpy()[0],
                name=f"image"
            )

            # Laser
            viewer.add_image(
                laser_tensor[0].detach().cpu().numpy(),
                name=f"laser"
            )

            # Labels
            colors = [
                (0.00, 0.0, 0.0, 1),
                (0.50, 0.0, 0.0, 1),
                (0.55, 0.0, 0.0, 1),
                (0.60, 0.0, 0.0, 1),
                (0.65, 0.0, 0.0, 1),
                (0.70, 0.0, 0.0, 1),
                (0.75, 0.0, 0.0, 1),
                (0.80, 0.0, 0.0, 1),
                (0.85, 0.0, 0.0, 1),
                (0.90, 0.0, 0.0, 1),
                (0.20, 0.7, 0.2, 1),
            ]
            for l in range(1, tumor_label-1):
                labels = batch["labels"][0][0].detach().cpu().numpy().astype(np.int32)
                labels = np.where(labels == l, 1, 0).astype(np.int32)

                viewer.add_labels(
                    labels,
                    name=f"{l}",
                    color={1: colors[l]}
                )

            # tumor (cleaned up)
            viewer.add_labels(
                tumor,
                name=f"tumor",
                color={1: colors[tumor_label]}
            )

            napari.run()


if __name__ == '__main__':
    main()
