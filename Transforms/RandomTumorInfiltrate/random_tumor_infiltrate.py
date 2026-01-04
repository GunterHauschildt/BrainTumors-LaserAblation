import monai
import nibabel as nib
import random
import torch
from Transforms.utils import TumorInfo
import skimage
import numpy as np
from scipy.ndimage import distance_transform_edt


def insert_tumor(
    image: np.ndarray,
    labels: np.ndarray,
    tumor: np.ndarray,
    tumor_mask: np.ndarray,
    brain_mask: np.ndarray,
    channel_num: int
):
    """
    image      : large 3D volume (Z, Y, X)
    tumor      : tumor intensity ROI
    tumor_mask : binary mask (same shape)
    center     : (z, y, x) insertion point in image
    feather    : softness in voxels
    """

    assert tumor.shape == tumor_mask.shape

    if tumor.shape[0] > image.shape[0] or tumor.shape[1] > image.shape[1] or tumor.shape[2] > image.shape[2]:
        return image, labels

    border0 = tumor.shape[0] // 4
    border1 = tumor.shape[1] // 4
    border2 = tumor.shape[2] // 4

    rng0_s = border0 + (tumor.shape[0] // 2)
    rng0_e = image.shape[0] - border0 - (tumor.shape[0] // 2)
    rng1_s = border1 + tumor.shape[1] // 2
    rng1_e = image.shape[1] - border1 - (tumor.shape[1] // 2)
    rng2_s = border2 + tumor.shape[2] // 2
    rng2_e = image.shape[2] - border2 - (tumor.shape[2] // 2)

    if rng0_s > rng0_e or rng1_s > rng1_e or rng2_s > rng2_e:
        return image, labels

    tries = 0
    while tries < 10:

        try:
            center = (
               random.randint(rng0_s, rng0_e),
               random.randint(rng1_s, rng1_e),
               random.randint(rng2_s, rng2_e)
            )

            # --- compute bounds ---
            s = center[0] - (tumor.shape[0] // 2), center[1] - (tumor.shape[1] // 2), center[2] - (tumor.shape[2] // 2)
            e = s[0] + tumor.shape[0], s[1] + tumor.shape[1], s[2] + tumor.shape[2]

            # --- is brain
            is_brain = brain_mask[s[0]:e[0], s[1]:e[1], s[2]:e[2]] != 0
            is_tumor = tumor_mask != 0
            is_brain_and_tumor = np.logical_and(is_brain, is_tumor)
            if np.count_nonzero(is_brain_and_tumor) < np.count_nonzero(tumor_mask) // 4:
                tries += 1
                continue

            # region views
            img_roi = image[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
            lbl_roi = labels[s[0]:e[0], s[1]:e[1], s[2]:e[2]]

            # is_brain_and_tumor : boolean mask where tumor is pasted
            tumor_mask = is_brain_and_tumor.astype(np.bool_)

            # --- build soft alpha using distance transform
            dist_in = distance_transform_edt(tumor_mask)
            m = dist_in.max()
            if m > 0:
                dist_in = dist_in / m
            feather = random.uniform(0.01, 0.1)
            dist_in = np.power(dist_in, feather)

            # --- blend tumor into image
            img_roi[:] = (
                (dist_in * tumor) + ((1.-dist_in) * img_roi)
            )

            # --- labels stay hard, but only where tumor exists
            lbl_roi[tumor_mask] = channel_num
            labels[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = lbl_roi # torch.from_numpy(lbl_roi)

            # # --- insert the tumor
            # image[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = np.where(
            #     is_brain_and_tumor,
            #     tumor * 10.,
            #     image[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
            # )
            #
            # # --- insert the tumor labels
            # labels[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = torch.from_numpy(np.where(
            #     is_brain_and_tumor,
            #     channel_num,
            #     labels[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
            # ))
            break

        except Exception as e:
            print(f"Error inserting tumor: {e}")
            break

    return image, labels


class RandomTumorInfiltrate(monai.transforms.Transform):

    """
    MONAI transform to infiltrate the image & labels with a tumor (from brats).
    """

    def __init__(self, tumor_channel, find_tumors, tumor_folder, tumor_modality, min_size=100):
        self._tumor_paths = find_tumors(tumor_folder, tumor_modality)
        self._tumor_channel = tumor_channel
        self._min_size = min_size
        pass

    def __call__(self, data: dict[str, np.ndarray | torch.Tensor]):
        d = dict(data)
        assert ("image" in data and "labels" in data and "zooms" in data)

        image = d["image"]
        if not isinstance(image, np.ndarray):
            image = image.detach().cpu().numpy()
        image_labels = d["labels"]
        if not isinstance(image_labels, np.ndarray):
            image_labels = image_labels.detach().cpu().numpy()
        image_zooms = d["zooms"]

        while True:
            tumor_paths = random.choice(self._tumor_paths)
            tumor_info = TumorInfo.load(tumor_paths.json_file)
            tumor_mask_np = np.load(tumor_paths.mask_np)
            tumor_image_np = np.load(tumor_paths.image_np)
            if (np.count_nonzero(tumor_mask_np)) > self._min_size:
                break

        # resize the tumor, so it's the same scale as the image.
        # they should be close enough that when we randomize later this step is in the noise.
        scale = np.array(image_zooms) / np.array(tumor_info.zooms)
        tumor_shape_p = tuple(int(round(s * f)) for s, f in zip(tumor_mask_np.shape, scale))

        def resize(m, shape_p, is_label):
            return skimage.transform.resize(
                m,
                shape_p,
                order=0 if is_label else 1,
                preserve_range=True,
                anti_aliasing=not is_label
            ).astype(m.dtype)

        tumor_image_np = resize(tumor_image_np, tumor_shape_p, False)
        tumor_mask_np = resize(tumor_mask_np, tumor_shape_p, True)

        for axis in [0, 1, 2]:
            if random.choice([True, False]):
                tumor_image_np = np.flip(tumor_image_np, axis=axis)
                tumor_mask_np = np.flip(tumor_mask_np, axis=axis)
        for axes in [(0, 1), (1, 2), (0, 2)]:
            if random.choice([True, False]):
                tumor_image_np = np.rot90(tumor_image_np, axes=axes)
                tumor_mask_np = np.rot90(tumor_mask_np, axes=axes)

        d["image"], d["labels"] = insert_tumor(
                image,
                image_labels,
                tumor_image_np,
                tumor_mask_np,
                d["brain"],
                self._tumor_channel
            )

        return d
