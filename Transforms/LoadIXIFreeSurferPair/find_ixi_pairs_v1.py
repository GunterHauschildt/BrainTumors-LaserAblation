import glob
from pathlib import Path


def find_ixi_pairs_v1(
    nifti_dir="F:/BrainTumors/IXI/IXI-T1",
    seg_dir="F:/BrainTumors/IXI/IXI-T1-segmented"
):
    """
    Finds pairs of (image, label) for IXI NIfTI and FreeSurfer MGZ segmentations.

    Returns:
        List[dict]:
            [
                {"image": "/path/to/file.nii.gz",
                 "label": "/path/to/aparc.DKTatlas+aseg.deep.mgz"},
                ...
            ]
    """
    nifti_dir = Path(nifti_dir)
    seg_dir = Path(seg_dir)

    # Find all nifti files
    nii_files = sorted(
        list(nifti_dir.glob("*.nii")) +
        list(nifti_dir.glob("*.nii.gz"))
    )

    # Build dictionary: key = subject ID, value = label path
    # Look for: */patient_SUBJECT/mri/aparc.DKTatlas+aseg.deep.mgz
    seg_paths = {}
    pattern = str(seg_dir / "*" / "patient_*" / "mri" / "aparc.DKTatlas+aseg.deep.mgz")

    for mgz in glob.glob(pattern):
        mgz = Path(mgz)

        # Extract subject between "patient_" and next slash
        patient_folder = mgz.parent.parent  # .../patient_SUBJECT
        name = patient_folder.name          # "patient_IXI123"
        if name.startswith("patient_"):
            subject_id = name[len("patient_"):]
        else:
            continue

        seg_paths[subject_id] = str(mgz)

    # Now pair NIfTI with matching segmentation
    items = []
    for nii in nii_files:
        subject_id = nii.stem  # e.g., "IXI123"
        if subject_id in seg_paths:
            items.append({
                "image": str(nii),
                "labels": seg_paths[subject_id]
            })

    return items

import glob
from pathlib import Path


def find_ixis_v1(
    nifti_dir="F:/BrainTumors/IXI/IXI-T1"
):
    nifti_dir = Path(nifti_dir)

    # Find all nifti files
    nii_files = sorted(
        list(nifti_dir.glob("*.nii")) +
        list(nifti_dir.glob("*.nii.gz"))
    )

    items = []
    for nii_file in nii_files:
        items.append({
            "image": str(nii_file)
        })

    return items

