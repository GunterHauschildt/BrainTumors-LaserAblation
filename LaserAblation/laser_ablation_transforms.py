from pathlib import Path

import monai
import torch

from Transforms.RelabelFreeSurfer.relabel_transform import RelabelFastSurferByName
from Transforms.LoadIXIFreeSurferPair.load_ixi_freesurfer_pair import LoadIXIFreeSurferPair, LoadNifti
from Transforms.ExtractBrainAndSkull.extract_brain_and_skull import ExtractBrainAndSkull
from Transforms.RandomTumorInfiltrate.random_tumor_infiltrate import RandomTumorInfiltrate
from Transforms.TumorAndStructureSegment.tumor_and_structure_segment import SegmentTumorAndStructureTransform
from Transforms.utils import TumorSamplePaths

import numpy as np
import json


fs_lut_path = "//FreeSurferColorLUT.txt"
mapping_by_name_path = "//Transforms/RelabelFreeSurfer/free_surfer_relabel.json"
tumor_path = "F:/BrainTumors/BraTS2021_tumors"


def find_tumors(root: str | Path, modality: str) -> list[TumorSamplePaths]:
    root = Path(root)
    samples: list[TumorSamplePaths] = []

    for tumor_dir in root.rglob("*"):
        if not tumor_dir.is_dir():
            continue

        tumor_info_json = tumor_dir / "tumor_info.json"
        mask_np = tumor_dir / "mask.npy"
        image_np = tumor_dir / ("tumor_" + modality + ".npy")

        if tumor_info_json.exists() and mask_np.exists() and mask_np.exists() and image_np.exists():
            samples.append(
                TumorSamplePaths(
                    json_file=tumor_info_json,
                    mask_np=mask_np,
                    image_np=image_np
                )
            )

    return samples


class ConstructRewardImage(monai.transforms.Transform):
    def __init__(self, weights: dict[int, float]):
        # self._weights = torch.zeros(max(weights.keys()) + 1, dtype=torch.float32, device="cuda")
        # for k, v in weights.items():
        #    self._weights[k] = v
        self._weights = weights

    def __call__(self, data):
        labels = data["labels"]
        reward = torch.zeros_like(labels, dtype=torch.float32, device="cuda")
        for lbl, w in self._weights.items():
            reward = torch.where(labels == lbl, w, reward)
        data["reward"] = reward
        return data


class LaserAblationTransform:
    reward_weights = {
        0: -0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
        8: 0.0,
        9: 0.0,
        10: +10.0
    }

    def __init__(self, config_file_name):

        try:
            with open(config_file_name) as config_json_file:
                config_json = json.load(config_json_file)
        except Exception as e:
            print("Error loading config file: ", e)
            exit(-1)

        self._input_shape = config_json['model']['input_shape']
        self._output_channels = config_json['model']['channels_out']

        pre_transforms = [

            # find the ixi / free-surfer labeled pairs.
            LoadIXIFreeSurferPair(self._input_shape),

            # find the skull and brain for future use.
            ExtractBrainAndSkull(fs_lut_path),

            # insert a tumor.
            RandomTumorInfiltrate(
                tumor_channel=None,
                find_tumors=find_tumors,
                tumor_folder=tumor_path,
                tumor_modality="t1",
                min_size=1000
            ),

            # scale the intensity.
            # note that we scale AFTER insert the tumor, with the intent being that this
            # should help/force the model to learn geometry anomalies, not just mri level.
            monai.transforms.ScaleIntensityRangePercentilesD(
                keys=["image"],
                lower=1,
                upper=99,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),

            # do a forward pass
            SegmentTumorAndStructureTransform(config_file_name),

            # un-one-hot
            monai.transforms.LambdaD(keys=["labels"], func=lambda x: x.argmax(axis=0)),
            # monai.transforms.LambdaD(keys=["labels"], func=lambda x: x.cuda()),

            # from labels to reward image
            ConstructRewardImage(LaserAblationTransform.reward_weights),

            # speak 'monai' from here on
            monai.transforms.LambdaD(keys=["image", "labels"], func=lambda x: x.unsqueeze(axis=0)),
        ]

        tensor_transforms = [
            monai.transforms.ToTensorD(keys=["image", "labels", "reward"], allow_missing_keys=True),
            monai.transforms.LambdaD(keys=["image", "labels", "reward"], allow_missing_keys=True, func=lambda x: x.cuda())
        ]

        augment_transforms = [

            # monai.transforms.RandAffineD(
            #     keys=['image', 'labels'],
            #     mode=('bilinear', 'nearest'),
            #     prob=0.50,
            #     rotate_range=(np.pi / 30., np.pi / 30., np.pi / 30.),
            #     translate_range=(8, 8, 8),
            #     scale_range=(0.10, 0.10, 0.10),
            #     padding_mode=monai.utils.GridSamplePadMode.BORDER,
            #     lazy=True,
            #     # spatial_size=self._roi_shape
            # ),
      ]

        transforms = pre_transforms + tensor_transforms
        self._valid_transforms = monai.transforms.Compose(transforms)
        self._test_transforms = monai.transforms.Compose(transforms)

        train_transforms = pre_transforms + augment_transforms + tensor_transforms
        self._train_transforms = monai.transforms.Compose(train_transforms)

    def train_transforms(self) -> monai.transforms.transform:
        return self._train_transforms

    def valid_transforms(self) -> monai.transforms.transform:
        return self._valid_transforms

    def test_transforms(self) -> monai.transforms.transform:
        return self._test_transforms

