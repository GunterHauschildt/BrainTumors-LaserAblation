import random
import cv2 as cv
from typing import Sequence, Dict, Optional
import glob
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt

def object(sz_xy: tuple[int, int], t: int = 3):
    w = sz_xy[0]
    h = sz_xy[1]
    msk = np.zeros([*sz_xy[::-1]]).astype(np.uint8)
    obj = np.zeros([*sz_xy[::-1], 3]).astype(np.uint8)
    r = 1
    while True:
        color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
        msk = cv.circle(msk, (w // 2, h // 2), r, (255,), t)
        obj = cv.circle(obj, (w // 2, h // 2), r, color, t)
        if np.count_nonzero(msk) == w * h:
            return obj
        r += t


def all_morphs(m: np.ndarray) -> Optional[tuple[int, dict[int, np.ndarray]]]:
    def squeeze_contour(c):
        return np.squeeze(c, axis=1)

    def squeeze_contours(cs):
        return [squeeze_contour(c) for c in cs]

    def stack_contours(cs: Sequence[np.ndarray]) -> np.ndarray:
        return np.vstack(cs)

    def find_contours(m: np.ndarray):
        contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE, None, None,
                                      offset_xy)
        if len(contours) == 0:
            return None, None, None

        contours = squeeze_contours(contours)
        all = stack_contours(contours)
        largest = max(contours, key=lambda c: c.shape[0])
        return contours, all, largest

    # def to_tensor(c):
    #    # c = np.stack(c)
    #    return torch.from_numpy(c).cuda()

    offset_xy = (-m.shape[1], -m.shape[0])
    # offset_xy = (0, 0)
    grow_xy = (-offset_xy[0], -offset_xy[1])

    m = cv.copyMakeBorder(m, grow_xy[1], grow_xy[1], grow_xy[0], grow_xy[0], cv.BORDER_CONSTANT,
                          None, (0, 0, 0))
    # cv.imshow("COPY MAKE", m)
    # cv.waitKey()

    morph_contours = {}

    contours, all, largest = find_contours(m)
    if contours is None:
        return None

    _, _, w, h = cv.boundingRect(largest)
    r = min(h, w) // 2
    morph_contours[r] = all

    rp = r - 1
    mp = m.copy()
    while True:
        mp = cv.morphologyEx(mp, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        if np.count_nonzero(mp) == 0:
            break

        contours, all, largest = find_contours(mp)
        if contours is None:
            break

        _, _, w, h = cv.boundingRect(largest)
        if min(w, h) == 1:
            break

        rp -= 1

    rp = r + 1
    mp = m.copy()
    while True:
        mp = cv.morphologyEx(mp, cv.MORPH_DILATE,
                             cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))

        contours, all, largest = find_contours(mp)

        if contours is None:
            break

        morph_contours[rp] = all

        rp += 1

        if np.count_nonzero(mp) == mp.shape[0] * mp.shape[1]:
            break

    return r, morph_contours


def get_nearest_point(contour: np.ndarray, p0: np.ndarray) -> tuple[int, int, int]:
    dif = contour - p0
    d2 = np.sum(dif * dif, axis=1)

    i = int(np.argmin(d2))
    x, y = contour[i]

    return i, x, y


# def tumor_slice(H, W, h, w, ch, cw, channels=1):
#
#     if channels is None or channels == 1:
#         color = (255, 255, 255)
#         m = np.zeros([H, W]).astype(np.uint8)
#     else:
#         color = (255, )
#         m = np.zeros([H, W, ch]).astype(np.uint8)
#
#     return cv.ellipse(m, (ch, cw), (h, w), 0., 0., 360., color, cv.FILLED)


def squish_1d_x_y(x, x0, x1, y0, y1):
    t = (x - x0) / (x1 - x0)
    t = max(0, min(1, t))
    return y0 + t * (y1 - y0)


def squish_1d_y_x(y, x0, x1, y0, y1):
    t = (y - y0) / (y1 - y0)
    t = max(0, min(1, t))
    return x0 + t * (x1 - x0)


def remap_1d(x0, x1, x2, y0, y1, y2) -> list[tuple[float, int]]:
    map_1d: list[tuple[float, int]] = []

    for y in range(y0, y1):
        x = squish_1d_y_x(y, x0, x1, y0, y1)
        map_1d.append((x, y))

    for y in range(y1, y2):
        x = squish_1d_y_x(y, x1, x2, y1, y2)
        map_1d.append((x, y))

    return map_1d



def pad_contour(contour, K, pad_value=1e9):
    """
    contour: (M,1,2) or (M,2)
    K: target number of points
    pad_value: value to fill padded entries
    """
    # Normalize shape to (M,2)
    if contour.ndim == 3:
        pts = contour[:,0,:]
    else:
        pts = contour

    M = pts.shape[0]

    out = np.full((K, 2), pad_value, dtype=pts.dtype)

    # Truncate or fill
    out[:min(M, K)] = pts[:min(M, K)]

    return out




def nearest_points(contour1, contour2):
    """
    contour1: (N,2)
    contour2: (M,2)
    Returns:
        indices: (N,) index in contour2 closest to each contour1 point
        dists:   (N,) distance values
    """

    diff = contour1[:, None, :] - contour2[None, :, :]  # (N,M,2)
    dist = np.sum(diff*diff, axis=2)  # (N,M)
    return np.argmin(dist, axis=1)


def fill_map(contour_i, contour_o, nearest, map_yx):

    Y, X = map_yx.shape[:2]
    for i, n in enumerate(nearest):
        xo, yo = contour_i[i]
        xi, yi = contour_o[n]
        if 0 <= xo < X and 0 <= yo < Y and map_yx[yo, xo, 0] == -1.:
            map_yx[yo, xo] = np.array([xi, yi])
    return map_yx


def compute_slice_displacement(
        slice_mask: np.ndarray,
        # map_1d: list[tuple[float, int]],
        size: tuple[int, int]
    ) -> np.ndarray:

    Y, X = size
    R = min(Y, X)

    r: int
    morph_contours: Dict[int, np.ndarray]
    r, morph_contours = all_morphs(slice_mask)

    if not morph_contours:
        return np.full([Y, X, 2], -1., dtype=np.float32)

    # 1D radius remap
    # r = min(h, w)
    # R = min(H, W)
    # map_1d = remap_1d(r // 2, r, R + R, r, r + r // 2, R + R)

    map_1d = remap_1d(r // 2, r, R + R, r, r + r // 2, R + R)

    # Empty displacement map
    map_yx = np.full([Y, X, 2], -1., dtype=np.float32)

    # Build displacement map
    pairs = []
    for map_r in map_1d:
        ro = map_r[1]
        ri = round(map_r[0])
        if ro not in morph_contours or ri not in morph_contours:
            continue

        for po in morph_contours[ro]:
            xo, yo = po
            _, xi, yi = get_nearest_point(morph_contours[ri], po)
            if 0 <= xo < X and 0 <= yo < Y and map_yx[yo, xo, 0] == -1.:
                map_yx[yo, xo] = np.array([xi, yi])
        # Stop early if corners are filled
        corners = np.array([map_yx[0, 0], map_yx[0, -1], map_yx[-1, 0], map_yx[-1, -1]])
        if np.all(corners != np.array([-1., -1.])):
            break

        # try the batched version
        pairs = nearest_points(morph_contours[ri], morph_contours[ro])
        fill_map(morph_contours[ri], morph_contours[ro], pairs, map_yx)
        corners = np.array([map_yx[0, 0], map_yx[0, -1], map_yx[-1, 0], map_yx[-1, -1]])
        if np.all(corners != np.array([-1., -1.])):
            break

        pass

    # Smooth the map
    def blur_size(dim, dim_perc=0.1):
        size = round(dim * dim_perc)
        return size if size % 2 else size + 1

    blur_y, blur_x = blur_size(Y), blur_size(X)
    map_yx = cv.blur(map_yx, (blur_x, blur_y), dst=None, anchor=(-1, -1),
                     borderType=cv.BORDER_REPLICATE)

    return map_yx


def pole_of_inaccess(mask3d: np.ndarray) -> tuple[int, int, int]:
    dist = distance_transform_edt(mask3d)
    z, y, x = np.unravel_index(np.argmax(dist), mask3d.shape)
    return int(z), int(y), int(x)


def pad_to_center(mask: np.ndarray, mask_center: Sequence, out_size: Sequence) -> np.ndarray:

    top = (out_size[0] // 2) - mask_center[0]
    bottom = out_size[0] - mask.shape[0] - top
    left = (out_size[1] // 2) - mask_center[1]
    right = out_size[1] - mask.shape[1] - left

    padded = cv.copyMakeBorder(
        mask,
        top, bottom, left, right,
        borderType=cv.BORDER_CONSTANT,
        dst=None,
        value=(0, 0, 0)
    )
    return padded


def robust_norm(slice_2d):
    s = slice_2d.astype(np.float32)
    pa, pb = np.percentile(s, (0, 100))
    s = np.clip(s, pa, pb)
    s = s - pa
    s = s / (pb - pa + 1e-5)
    s = (s * 255).astype(np.uint8)
    return s


if __name__ == "__main__":

    # Load the test image.
    # To do: we're assuming the Z,Y,X are constant for everything in IXI

    # # test_path = "C:/Users/gunte/OneDrive/Desktop/BraTS-small/IXI/IXI002-Guys-0828-T1.nii.gz"
    # #
    # # test_img = nib.load(test_path)
    # # test_img = nib.as_closest_canonical(test_img, enforce_diag=False)
    # # test_data = test_img.get_fdata().astype(np.uint8)
    # # test_affine = test_img.affine
    # # axcodes = nib.aff2axcodes(test_affine)
    # #
    # # print(f"{axcodes}")
    # # print(f"Zooms: {test_img.header.get_zooms()}")
    # # print(f"Shape: {test_data.shape}")
    # # print(f"Affine: {test_affine}")
    #
    # # exit(0)
    #
    # if axcodes[0] == 'R' and axcodes[1] == 'A' and axcodes[2] == 'S':
    #     X, Y, Z = test_data.shape
    # else:
    #     raise "I'm the wrong axis order."

    X = 256 # 150
    Y = 256
    Z = 256
    test_data = object((Y, X))
    map_size = (X, Y)
    print(f"Volume shape: Z={Z}, Y={Y}, X={X}")

    # test_data = robust_norm(test_data)
    # for z in range(Z):
    #     brain = test_data[:, :, z]
    #
    #     cv.imshow("slice", brain)
    #     cv.waitKey()

    # iterate over all the tumors
    tumor_root = "c:/Users/gunte/OneDrive/Desktop/BraTS-small/braTS2021_tumors"
    mask_files = sorted(glob.glob(os.path.join(tumor_root, "**", "mask.nii.gz"), recursive=True))
    print(f"Found {len(mask_files)} tumor masks.")

    for mask_path in mask_files:
        subject_dir = os.path.dirname(mask_path)
        subject_name = os.path.basename(subject_dir)
        print(f"\n=== Processing {subject_name} ===")

        # Load the mask in RAS canonical orientation
        mask_img = nib.load(mask_path)
        mask_img = nib.as_closest_canonical(mask_img)
        mask_data = mask_img.get_fdata().astype(np.uint8)
        axcodes = nib.aff2axcodes(mask_img.affine)

        print(f"{axcodes}")
        print(f"Zooms: {mask_img.header.get_zooms()}")
        print(f"Shape: {mask_data.shape}")
        print(f"Affine: {mask_img.affine}")

        if axcodes[0] == 'R' and axcodes[1] == 'A' and axcodes[2] == 'S':
            x_tumor, y_tumor, z_tumor = mask_data.shape
        else:
            raise "I'm the wrong axis order."

        # Get volume dimensions
        print(f"Tumor shape: Z={z_tumor}, Y={y_tumor}, X={x_tumor}")

        center_xyz = pole_of_inaccess(mask_data)

        # -------------------------------
        # 3. Iterate over Z slices
        # -------------------------------
        for z in range(z_tumor):

            brain = test_data # test_data[:, :, z + 50] # ToDo: I made this up!!
            # brain = cv.cvtColor(brain, cv.COLOR_GRAY2BGR)

            slice_mask = mask_data[:, :, z] * 255
            slice_mask = pad_to_center(slice_mask, center_xyz[:2], (X, Y))

            if np.count_nonzero(slice_mask) == 0:
                continue

            cv.imshow(f"tumor_slice_mask", slice_mask)
            cv.waitKey(1)

            map_yx = compute_slice_displacement(slice_mask, map_size)

            warped = cv.remap(
                brain, map_yx[:, :, 0], map_yx[:, :, 1],
                interpolation=cv.INTER_LINEAR,
                borderMode=cv.BORDER_REFLECT
            )

            warped[slice_mask != 0] = (0, 0, 0)

            cv.imshow("BRAIN", brain)
            cv.imshow("WARPED", warped)
            cv.waitKey()

            # out_file = os.path.join(subject_out, f"displacement_z{z:03d}.npy")
            # np.save(out_file, map_yx)
            # print(f"  Saved slice {z} displacement map: {out_file}")

        # Optional: save per-slice displacement maps to disk
        # np.save(os.path.join(subject_dir, f"displacement_slice_{z:03d}.npy"), displacement_map_slice)
