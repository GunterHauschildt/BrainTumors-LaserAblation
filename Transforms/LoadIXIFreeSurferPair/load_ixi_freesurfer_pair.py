import monai
import nibabel as nib
import numpy as np


class LoadIXIFreeSurferPair(monai.transforms.Transform):

    """
    MONAI transform to load IXI and its corresponding FreeSurfer segmentation
    (which needs resampling, re-spacing).
    """

    def __init__(self, expected_input_shape: tuple[int, int, int]):
        self._expected_input_shape = expected_input_shape
        pass

    def __call__(self, data):
        d = dict(data)
        out = {}

        # load image
        image = nib.load(d["image"])
        image = nib.as_closest_canonical(image)
        image_data = image.get_fdata().astype(np.float32)
        image_affine = image.affine

        # load labels
        labels = nib.load(d["labels"])
        labels = nib.as_closest_canonical(labels)
        labels_data = labels.get_fdata().astype(np.int16)
        labels_affine = labels.affine

        # monai transforms want a batch
        image_data = np.expand_dims(image_data, axis=0)
        labels_data = np.expand_dims(labels_data, axis=0)

        # resample in size so freesurfer labels match the original image
        labels_data = monai.transforms.Spacing(
            pixdim=image.header.get_zooms(),
            mode="nearest"
        )(monai.data.MetaTensor(labels_data, affine=labels_affine))

        # crop labels (to fit the original image size)
        roi_start = []
        roi_end = []
        for axis in range(1, 4):  # H, W, D
            diff = labels_data.shape[axis] - image_data.shape[axis]
            if diff > 0:
                # crop symmetrically
                start = diff // 2
                end = start + image_data.shape[axis]
            else:
                # no crop needed
                start = 0
                end = labels_data.shape[axis]
            roi_start.append(start)
            roi_end.append(end)

        labels_data = monai.transforms.SpatialCrop(
            roi_start=tuple(roi_start),
            roi_end=tuple(roi_end)
        )(labels_data)

        # and pad if necessary
        image_data = monai.transforms.SpatialPad(
            self._expected_input_shape,
            method='symmetric'
        )(image_data)
        labels_data = monai.transforms.SpatialPad(
            self._expected_input_shape,
            method='symmetric'
        )(labels_data)

        out["affine"] = image_affine
        out["zooms"] = image.header.get_zooms()
        out["image"] = np.squeeze(image_data, axis=0)
        out["labels"] = np.squeeze(labels_data, axis=0)
        return out


# class LoadIXI(monai.transforms.Transform):
#     def __init__(self, expected_input_shape: tuple[int, int, int]):
#         self._expected_input_shape = expected_input_shape
#         pass
#
#     def __call__(self, data):
#         # load image
#         image = nib.load(data["image"])
#         image = nib.as_closest_canonical(image)
#         image_data = image.get_fdata().astype(np.float32)
#         image_affine = image.affine
#
#         # monai transforms want a batch
#         image_data = np.expand_dims(image_data, axis=0)
#
#         image_data = monai.transforms.SpatialPad(
#             self._expected_input_shape,
#             method='symmetric'
#         )(image_data)
#
#         data["affine"] = image_affine
#         data["zooms"] = image.header.get_zooms()
#         data["image"] = np.squeeze(image_data, axis=0)
#         return data
#

class LoadNifti(monai.transforms.Transform):
    def __init__(self, expected_input_shape: tuple[int, int, int]):
        self._expected_input_shape = expected_input_shape
        pass

    def __call__(self, data):
        # load image
        image = nib.load(data["image_path"])
        image = nib.as_closest_canonical(image)
        image_data = image.get_fdata().astype(np.float32)
        image_affine = image.affine

        # monai transforms want a batch
        image_data = np.expand_dims(image_data, axis=0)

        image_data = monai.transforms.SpatialPad(
            self._expected_input_shape,
            method='symmetric'
        )(image_data)

        data["affine"] = image_affine
        data["zooms"] = image.header.get_zooms()
        data["image"] = np.squeeze(image_data, axis=0)
        return data
