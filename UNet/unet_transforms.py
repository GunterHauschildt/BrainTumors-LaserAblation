from pathlib import Path

import monai
from Transforms.RelabelFreeSurfer.relabel_transform import RelabelFastSurferByName
from Transforms.LoadIXIFreeSurferPair.load_ixi_freesurfer_pair import LoadIXIFreeSurferPair
from Transforms.ExtractBrainAndSkull.extract_brain_and_skull import ExtractBrainAndSkull
from Transforms.RandomTumorInfiltrate.random_tumor_infiltrate import RandomTumorInfiltrate
from Transforms.utils import TumorSamplePaths

import numpy as np

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


class IXI_Tumors_UNet_Transforms:
    def __init__(self,
                 input_shape: tuple[int, int, int],
                 roi_shape: None or tuple[int, int, int],
                 input_channels: int,
                 output_channels: int
                 ):

        self._input_shape = input_shape
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._roi_shape = roi_shape

        pre_transforms = [

            # find the ixi / free-surfer labeled pairs.
            LoadIXIFreeSurferPair(self._input_shape),

            # find the skull and brain for future use.
            ExtractBrainAndSkull(fs_lut_path),

            # relabel from the many freesurfer labels to just the ones we care about
            RelabelFastSurferByName(
                keys=["labels"],
                fs_lut_path=fs_lut_path,
                mapping_by_name_path=mapping_by_name_path
            ),

            # insert a tumor.
            RandomTumorInfiltrate(
                tumor_channel=10,
                find_tumors=find_tumors,
                tumor_folder=tumor_path,
                tumor_modality="t1"
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

            # speak 'monai' from here on
            monai.transforms.LambdaD(keys=["image", "labels"], func=lambda x: x[None]),
            monai.transforms.AsDiscreteD(keys=["labels"], to_onehot=self._output_channels),
        ]

        roi_transforms = [
            monai.transforms.RandSpatialCropD(
                keys=["image", "labels"],
                roi_size=self._roi_shape,
                random_size=False),
        ]

        tensor_transforms = [
            monai.transforms.ToTensorD(keys=["image", "labels"], allow_missing_keys=True)
        ]

        augment_transforms = [

            monai.transforms.RandAffineD(
                keys=['image', 'labels'],
                mode=('bilinear', 'nearest'),
                prob=0.50,
                rotate_range=(np.pi / 30., np.pi / 30., np.pi / 30.),
                translate_range=(8, 8, 8),
                scale_range=(0.10, 0.10, 0.10),
                padding_mode=monai.utils.GridSamplePadMode.BORDER,
                lazy=True,
                spatial_size=self._roi_shape
            ),

            monai.transforms.RandScaleIntensityD(
                keys=['image'],
                prob=.25,
                factors=.10
            ),

            monai.transforms.RandGaussianSmoothD(
                keys=['image'],
                prob=.10,
                sigma_x=(.95, 1.05),
                sigma_y=(.95, 1.05),
                sigma_z=(.95, 1.05)
            ),

            monai.transforms.RandGaussianSharpenD(
                keys=['image'],
                prob=.10,
                sigma1_x=(.90, 1.00),
                sigma1_y=(.90, 1.00),
                sigma1_z=(.90, 1.00),
                sigma2_x=.5,
                sigma2_y=.5,
                sigma2_z=.5,
                alpha=(1.0, 5.0)
            )
        ]

        transforms = pre_transforms + tensor_transforms
        self._valid_transforms = monai.transforms.Compose(transforms)
        self._test_transforms = monai.transforms.Compose(transforms)

        if self._roi_shape is None:
            train_transforms = pre_transforms + augment_transforms + tensor_transforms
        else:
            train_transforms = pre_transforms + roi_transforms + augment_transforms + tensor_transforms

        self._train_transforms = monai.transforms.Compose(train_transforms)

    def train_transforms(self) -> monai.transforms.transform:
        return self._train_transforms

    def valid_transforms(self) -> monai.transforms.transform:
        return self._valid_transforms

    def test_transforms(self) -> monai.transforms.transform:
        return self._test_transforms

