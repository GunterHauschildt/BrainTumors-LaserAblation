import random
import math
import cv2 as cv
import numpy as np
import ctypes
import nibabel as nib


class draw_nifti_info():
    def __init__(self, name, path, normalize=True):
        self.name = name
        self.path = path
        self.normalize = True


def draw_niftis(draw_nifti_infos):  #, superimpose):

    images_out = []
    for draw_nifti_info in draw_nifti_infos:
        name = draw_nifti_info.name
        path = draw_nifti_info.path
        normalize = draw_nifti_info.normalize

        image = nib.load(path).get_fdata()
        draw = draw_volume_and_segmentation(
            image,
            None,
            total=len(draw_nifti_infos),
            axis=0,
            rotate=0,
            normalize=normalize
        )
        images_out.append((draw, image.shape[0], image.shape[1], image.shape[2]))

    return images_out


    # images = {}
    # for draw_nifti_info in draw_nifti_infos:
    #     name = draw_nifti_info.name
    #     cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
    #     # cv.resizeWindow(name, win_size[0], win_size[1])
    #     cv.moveWindow(name, win_x, 250)
    #     path = draw_nifti_info.path
    #     image = nib.load(path).get_fdata()
    #     images[name] = (image, draw_nifti_info.normalize)
    #     win_x += (image.shape[1] + 50)
    #
    #     if Z is None:
    #         Z = image.shape[2]
    #
    # for z in range(Z):
    #     try:
    #         for image_name in images:
    #             image = images[image_name][0]
    #             normalize = images[image_name][1]
    #
    #             # img_draw = np.expand_dims(image[:, :, slice], axis=2).astype(dtype=np.uint8) * 255
    #             img_draw = np.expand_dims(image[:, :, z], axis=2)
    #             if normalize:
    #                 img_draw = cv.normalize(img_draw, None, 0, 255.0, cv.NORM_MINMAX).astype(dtype=np.uint8)
    #             # contours, _ = cv.findContours(ls_slice, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    #             img_draw = cv.cvtColor(img_draw, cv.COLOR_GRAY2BGR)
    #             cv.imshow(image_name, img_draw)
    #     except:
    #         pass
    #     cv.waitKey()

def draw_volume_and_segmentation(vol_array, seg_array, axis=0, rotate=0, normalize=False):

    if vol_array is None and seg_array is None:
        return

    if vol_array is None:
        vol_array = np.zeros((seg_array.shape[1], seg_array.shape[2], seg_array.shape[3]), dtype=np.float32)
    else:
        vol_array = vol_array.astype(dtype=np.float32)

    # resize on the input otherwise this can take gobs of memory (and break training)

    p_axis = (1, 2)
    if axis == 0:
        p_axis = (1, 2)
    elif axis == 1:
        p_axis = (0, 2)
    elif axis == 2:
        p_axis = (0, 1)

    sqrt_num_images = math.sqrt(vol_array.shape[axis])

    # because we draw 3
    grid_display_ratio = (1, .33)

    grid = (1, 1)
    monitor_resolution = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))
    grid_aspect_ratio = (monitor_resolution[0] * grid_display_ratio[0]) / (monitor_resolution[1] * grid_display_ratio[1])
    image_aspect_ratio = vol_array.shape[p_axis[0]] / vol_array.shape[p_axis[1]]
    image_cols = vol_array.shape[p_axis[0]]
    image_rows = vol_array.shape[p_axis[1]]
    if rotate == 90 or rotate == 180:
        image_aspect_ratio = 1.0 / image_aspect_ratio
        image_cols = vol_array.shape[p_axis[1]]
        image_rows = vol_array.shape[p_axis[0]]

    best_error = np.inf
    for col in range(1, vol_array.shape[axis]):
        min_grid_rows = math.ceil(vol_array.shape[axis] / col)
        max_grid_rows = min_grid_rows + 1
        for row in range(min_grid_rows, max_grid_rows):
            error = abs(grid_aspect_ratio - (col/row))
            if error < best_error:
                grid = (row, col)
                best_error = error

    # grid = (7, 37)
    resize_cols = round((monitor_resolution[0] * .8) / grid[1])
    resize_scale = resize_cols / image_cols
    resize_rows = round(image_rows * resize_scale)

    # colors = []
    # random.seed(2)
    # for c in range(0, 100):
    #     colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    colors = [
        (0, 0, 0),
        (0, 240, 0),
        (0, 220, 0),
        (0, 200, 0),
        (0, 180, 0),
        (0, 140, 0),
        (0, 120, 0),
        (0, 100, 0),
        (0, 80, 0),
        (0, 60, 0),
        (0, 0, 200)
    ]

    if 1:  #normalize:
        vol_array = cv.normalize(vol_array, None, 0.0, 1.0, cv.NORM_MINMAX)
    vol_array *= 255
    vol_array = np.clip(vol_array,0, 255).astype(dtype=np.uint8)

    img_grid = np.zeros((resize_rows * grid[0], resize_cols * grid[1], 3), dtype=np.uint8)

    img_draw = None
    segments = None
    for i in range(0, vol_array.shape[axis]):
        this_grid_r = i // grid[1]
        this_grid_c = i % grid[1]
        if axis == 0:
            img_draw = np.expand_dims(vol_array[i, :, :], axis=2)
            if seg_array is not None:
                segments = seg_array[ i, :, :]
        elif axis == 1:
            img_draw = np.expand_dims(vol_array[:, i, :], axis=2)
            if seg_array is not None:
                segments = seg_array[:, i, :]
        elif axis == 2:
            img_draw = np.expand_dims(vol_array[:, :, i], axis=2)
            if seg_array is not None:
                segments = seg_array[:, :, i]

        img_draw = cv.cvtColor(img_draw, cv.COLOR_GRAY2BGR)

        img_draw_largest = None
        if seg_array is not None:
            # segmentations = np.unique(segments.astype(np.uint8))
            # FIXME: 10 should be num labels
            for segmentation in range(0, 11):
                if segmentation == 0:
                    continue
                mask = np.where(segments == segmentation, 1, 0).astype(dtype=np.uint8)
                mask = np.expand_dims(mask, axis=2)
                contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                img_draw = cv.drawContours(img_draw, contours, -1, colors[segmentation], 3)

        img_draw = cv.resize(img_draw, (resize_cols, resize_rows))

        grid_y0 = this_grid_r * resize_rows
        grid_y1 = grid_y0 + resize_rows
        grid_x0 = this_grid_c * resize_cols
        grid_x1 = grid_x0 + resize_cols

        # to do I've got something wrong fix later, update think its fixed now though
        try:
            img_grid[grid_y0:grid_y1, grid_x0:grid_x1, :] = img_draw
        except:
            pass

    return img_grid


def draw_volume_and_segmentation_largest(vol_array, seg_array, axis=0, normalize=True):

    if vol_array is None and seg_array is None:
        return

    if vol_array is None:
        vol_array = np.zeros((seg_array.shape[0], seg_array.shape[1], seg_array.shape[2]), dtype=np.float32)
    else:
        vol_array = vol_array.astype(dtype=np.float32)

    if axis == 0:
        p_axis = (1, 2)
    elif axis == 1:
        p_axis = (0, 2)
    elif axis == 2:
        p_axis = (0, 1)

    colors = [
        (0, 0, 0),
        (0, 240, 0),
        (0, 220, 0),
        (0, 200, 0),
        (0, 180, 0),
        (0, 140, 0),
        (0, 120, 0),
        (0, 100, 0),
        (0, 80, 0),
        (0, 60, 0),
        (0, 0, 200)
    ]
    # random.seed(2)
    # for c in range(0, 100):
    #     colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    if normalize:
        vol_array = cv.normalize(vol_array, None, 0.0, 1.0, cv.NORM_MINMAX)
    vol_array *= 255
    vol_array = np.clip(vol_array, 0, 255).astype(dtype=np.uint8)

    img_draw = None
    img_largest = None
    segments = None
    max_contour_size = 0
    for i in range(0, vol_array.shape[axis]):
        if axis == 0:
            img_draw = np.expand_dims(vol_array[i, :, :], axis=2)
            if seg_array is not None:
                segments = seg_array[i, :, :]
        elif axis == 1:
            img_draw = np.expand_dims(vol_array[:, i, :], axis=2)
            if seg_array is not None:
                segments = seg_array[:, i, :]
        elif axis == 2:
            img_draw = np.expand_dims(vol_array[:, :, i], axis=2)
            if seg_array is not None:
                segments = seg_array[:, :, i]

        img_draw = cv.cvtColor(img_draw, cv.COLOR_GRAY2BGR)

        if seg_array is not None:
            segmentations = np.unique(segments.astype(np.uint8))
            for segmentation in segmentations:
                if segmentation == 0:
                    continue
                mask = np.where(segments == segmentation, 1, 0).astype(dtype=np.uint8)
                mask = np.expand_dims(mask, axis=2)
                contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

                contour_size = 0
                for contour in contours:
                    if contour.shape[0] > contour_size:
                        contour_size = contour.shape[0]

                if contour_size > max_contour_size:
                    max_contour_size = contour_size
                    img_largest = cv.drawContours(img_draw, contours, -1, colors[segmentation], 2)

    if img_largest is None:
        if axis == 0:
            img_largest = np.expand_dims(vol_array[vol_array.shape[0] // 2, :, :], axis=2)
        elif axis == 1:
            img_largest = np.expand_dims(vol_array[:, vol_array.shape[1] // 2, :], axis=2)
        elif axis == 2:
            img_largest = np.expand_dims(vol_array[:, :, vol_array.shape[0] // 2], axis=2)

    return img_largest



# def draw_dicom_files(folder_name, wait=-1, is_segmentation=False):
#     reader = sitk.ImageSeriesReader()
#     dicom_names = reader.GetGDCMSeriesFileNames(str(folder_name))
#     reader.SetFileNames(dicom_names)
#     volume = reader.Execute()
#     size = volume.GetSize()
#     data = sitk.GetArrayFromImage(volume)
#     if not is_segmentation:
#         draw = draw_volume_and_segmentation(data, None, 0)
#     else:
#         data = np.squeeze(data, axis=0)
#         data //= 255        # fix me, works for this case, but i should be getting 0,1,2,3,4 etc here, what if its more that one class?
#         max = np.max(data)
#         draw = draw_volume_and_segmentation(None, data, 0)
#
#     winname = str(folder_name)
#     cv.namedWindow(winname, cv.WINDOW_NORMAL)
#     cv.resizeWindow(winname, 512, 512)
#     cv.imshow(winname, draw)
#     if wait == -1:
#         wait = 1000
#     cv.waitKey(wait)
#     cv.destroyWindow(winname)
