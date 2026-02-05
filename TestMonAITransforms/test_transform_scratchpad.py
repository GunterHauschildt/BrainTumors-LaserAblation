import json
import nibabel as nib
import numpy as np
import monai
from Transforms.RelabelFreeSurfer.relabel_transform import RelabelFastSurferByName
from Transforms.LoadIXIFreeSurferPair.load_ixi_freesurfer_pair import LoadIXIFreeSurferPair
from Transforms.RandomTumorInfiltrate.random_tumor_infiltrate import RandomTumorInfiltrate
from Transforms.ExtractBrainAndSkull.extract_brain_and_skull import ExtractBrainAndSkull
from UNet.unet_transforms import find_tumors
from Transforms.LoadIXIFreeSurferPair.find_ixi_pairs_v1 import find_ixi_pairs_v1
import napari


# --- load your mapping JSON ---
# with open("../Transforms/RelabelFreeSurfer/free_surfer_relabel.json", "r") as f:
#     mapping_by_name = json.load(f)


# --- paths ---
fs_lut_path = "FreeSurferColorLUT.txt"
mapping_by_name_path = "Transforms/RelabelFreeSurfer/free_surfer_relabel.json"
image_path = "F:/BrainTumors/IXI/IXI-T1/IXI002-Guys-0828-T1.nii.gz"
label_path = "F:/BrainTumors/IXI/IXI-T1-segmented/IXI002-Guys-0828-T1.nii/patient_IXI002-Guys-0828-T1.nii/mri/aparc.DKTatlas+aseg.deep.mgz"
tumor_path = "F:/BrainTumors/BraTS2021_tumors"


# --- relabel
relabel_transform = RelabelFastSurferByName(
    keys=["labels"],
    fs_lut_path=fs_lut_path,
    mapping_by_name_path=mapping_by_name_path
)

random_tumor_infiltrate = RandomTumorInfiltrate(
    tumor_channel=10,
    find_tumors=find_tumors,
    tumor_folder=tumor_path,
    tumor_modality='flair'
)

add_channel = monai.transforms.LambdaD(keys=["image", "labels"], func=lambda x: x[None])

test_transforms = [
    LoadIXIFreeSurferPair((150, 256, 256)),
    ExtractBrainAndSkull(fs_lut_path),
    relabel_transform,
    random_tumor_infiltrate,
    add_channel,
    monai.transforms.AsDiscreteD(keys=["labels"], to_onehot=11)
]

test_transforms = monai.transforms.Compose(test_transforms)

ixi_pairs = find_ixi_pairs_v1("F:/BrainTumors/IXI/IXI-T1","F:/BrainTumors/IXI/IXI-T1-segmented")

for ixi_pair in ixi_pairs:

    data = test_transforms(ixi_pair)
    if isinstance(data["image"], np.ndarray):
        image = data["image"][0].astype(np.float32)
    else:
        image = data["image"][0].numpy().astype(np.float32)


    if isinstance(data["image"], np.ndarray):
        labels = data["labels"].astype(np.int16)
    else:
        labels = data["labels"].numpy().astype(np.int16)
    if isinstance(data["brain"], np.ndarray):
        brain = data["brain"].astype(np.int16)
    else:
        brain = data["brain"].numpy().astype(np.int16)

    viewer = napari.Viewer()

    # Image.
    viewer.add_image(
        image
    )

    # Brain
    viewer.add_labels(
        brain
    )

    # Remapped classes.
    colors = [
        (0.0, 0.0, 0, 1),
        (0.1, 0.0, 0, 1),
        (0.2, 0.0, 0, 1),
        (0.3, 0.0, 0, 1),
        (0.4, 0.0, 0, 1),
        (0.5, 0.0, 0, 1),
        (0.6, 0.0, 0, 1),
        (0.7, 0.0, 0, 1),
        (0.8, 0.0, 0, 1),
        (0.9, 0.0, 0, 1),
        (0.0, 1.0, 0, 1),
    ]

    for l in range(1, 11):
        viewer.add_labels(
            labels[l],
            name=f"{l}",
            color={1: colors[l]}
        )


    napari.run()
