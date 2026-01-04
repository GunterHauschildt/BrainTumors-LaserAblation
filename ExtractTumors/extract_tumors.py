from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence, List, Any
import numpy as np
import nibabel as nib
from numpy import ndarray
from skimage.measure import label
from scipy.ndimage import find_objects
import napari
import glob
from dataclasses import asdict
import json
from Transforms.utils import Point, BoundingBox3D, TumorInfo
from pathlib import Path


# ------------------------------------------------------------
# Bounding box helpers
# ------------------------------------------------------------

def masks_and_bounding_boxes(mask: np.ndarray) -> List[tuple[int, np.ndarray, BoundingBox3D]]:
    """Return list of BoundingBox3D for all connected components in a binary mask."""
    labeled_mask = label(mask)  # > 0)
    labels = np.unique(labeled_mask)
    labels = labels[labels != 0]
    masks_and_bboxes = []
    for l in labels:
        this_mask = (labeled_mask == l).astype(np.uint8)
        slices = find_objects(this_mask)
        for s in slices:
            if s is None:
                continue
            z0, z1 = s[0].start, s[0].stop
            y0, y1 = s[1].start, s[1].stop
            x0, x1 = s[2].start, s[2].stop
            masks_and_bboxes.append((l, this_mask, BoundingBox3D(
                p0=Point(x0, y0, z0),
                p1=Point(x1, y1, z1))
            ))
    return masks_and_bboxes


# ------------------------------------------------------------
# Affine adjustment
# ------------------------------------------------------------

def bounding_box_affine(affine: np.ndarray, bbox: BoundingBox3D) -> np.ndarray:
    """Shift affine after cropping."""
    R = affine[:3, :3]
    t = affine[:3, 3]
    offset = np.array([bbox.p0.z, bbox.p0.y, bbox.p0.x])
    new_affine = affine.copy()
    new_affine[:3, 3] = t + R @ offset
    return new_affine

# ------------------------------------------------------------
# Loading / saving helpers
# ------------------------------------------------------------

def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """Load a NIfTI file and force it to RAS canonical orientation."""
    img = nib.load(path)
    img_ras = nib.as_closest_canonical(img)
    data = img_ras.get_fdata(dtype=np.float32)
    affine = img_ras.affine
    zooms = img.header.get_zooms()
    return data, affine, zooms

def save_nifti(path: str, data: np.ndarray, affine: np.ndarray):
    nib.save(nib.Nifti1Image(data, affine), path)


def preview_subject_rois(
    subject_dir: str,
    output_folder: str,
    modality: str = "flair"
):

    subject_name = os.path.basename(subject_dir.rstrip("/\\"))
    print(f"\n=== Previewing subject {subject_name} ===")

    # Load full image (reference volume)
    full_path = os.path.join(subject_dir, f"{subject_name}_{modality}.nii.gz")
    full_data, full_affine, full_zooms = load_nifti(full_path)

    # Find ROI folders
    roi_dirs = sorted(glob.glob(os.path.join(output_folder, f"tumor*")))
    if not roi_dirs:
        print("No ROI directories found.")
        return

    print(f"Found {len(roi_dirs)} ROI directories")

    # Start viewer
    viewer = napari.Viewer()

    # Background image
    viewer.add_image(
        full_data,
        affine=full_affine,
        name=f"{subject_name}_{modality}",
        blending="additive"
    )

    # Add each ROI by loading *its own affine*
    for rd in roi_dirs:
        mask_path = os.path.join(rd, "mask.nii.gz")
        if not os.path.exists(mask_path):
            continue
        image_path = os.path.join(rd, "tumor.nii.gz")
        if not os.path.exists(image_path):
            continue

        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        mask_aff = mask_img.affine  # already correct (same world-space origin)

        image_img = nib.load(image_path)
        image_data = image_img.get_fdata()
        image_aff = image_img.affine  # already correct (same world-space origin)

        viewer.add_labels(
            mask_data.astype("uint8"),
            affine=mask_aff,
            name=f"Mask: {os.path.basename(rd)}"
        )

        viewer.add_image(
            image_data,
            affine=image_aff,
            name=f"Tumor: {os.path.basename(rd)}"
        )

    napari.run()


# ------------------------------------------------------------
# Extract whole tumor from a single BraTS subject
# ------------------------------------------------------------

def extract_brats_tumors(
    subject_dir: str,
    output_dir: str
):

    patient_name = os.path.basename(subject_dir)

    # Load mask & create whole tumor mask.
    labels, affine, zooms = load_nifti(
        os.path.join(subject_dir, f"{patient_name}_seg.nii.gz")
    )
    whole_tumor_mask = (labels > 0)
    # find tumors
    tumor_masks_and_bboxes = masks_and_bounding_boxes(whole_tumor_mask)

    suffixes = ["t1", "t1ce", "t2", "flair"]
    images = {}
    for suffix in suffixes:
        images[suffix], _, _ = load_nifti(
            os.path.join(subject_dir, f"{patient_name}_{suffix}.nii.gz")
        )

    for i, (_, mask, bbox) in enumerate(tumor_masks_and_bboxes):

        bbox_affine = bounding_box_affine(affine, bbox)
        bbox_mask = mask[bbox.as_slices()].astype(np.uint8)
        bbox_tumors = {}
        for suffix in suffixes:
            bbox_tumors[suffix] = np.where(bbox_mask > 0, images[suffix][bbox.as_slices()], 0.)

        out_sub = os.path.join(output_dir, f"tumor{i:02d}")
        os.makedirs(out_sub, exist_ok=True)
        save_nifti(os.path.join(out_sub, f"mask.nii.gz"), bbox_mask, bbox_affine)
        np.save(os.path.join(out_sub, "mask.npy"), bbox_mask)
        for suffix in suffixes:
            save_nifti(
                os.path.join(out_sub, f"tumor_{suffix}.nii.gz"), bbox_tumors[suffix], bbox_affine
            )
            np.save(os.path.join(out_sub, f"tumor_{suffix}.npy"), bbox_tumors[suffix])
        tumor_info = TumorInfo(
            patient_name,
            zooms,
            bbox_affine,
            bbox
        )
        tumor_info.save(Path(out_sub))


if __name__ == "__main__":
    brats_root = "F:/BrainTumors/BraTS2021"
    tumor_root = "F:/BrainTumors/BraTS2021_tumors"
    os.makedirs(tumor_root, exist_ok=True)

    cases = sorted([
        os.path.join(brats_root, d)
        for d in os.listdir(brats_root)
        if d.startswith("BraTS2021_") and os.path.isdir(os.path.join(brats_root, d))
    ])

    print(f"Found {len(cases)} BRATS cases.")

    total_tumors = 0
    for case_path in cases:
        case_id = os.path.basename(case_path)
        print(f"\n=== Processing {case_id} ===")
        try:
            extract_brats_tumors(
                case_path,
                os.path.join(tumor_root, case_id)
            )
        except Exception as e:
            print(f" !! Failed on {case_id}: {e}")

        # preview_subject_rois(case_path, os.path.join(tumor_root, case_id), modality="flair")

    print(f"\nDone. Total extracted tumor ROIs: {total_tumors}")
