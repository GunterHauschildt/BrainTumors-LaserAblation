import random
import math
import cv2 as cv
import numpy as np

from UNet.drawing_utils import *


def set_windows():
    cv.namedWindow("CT (slices)", cv.WINDOW_NORMAL)
    cv.namedWindow("GT (slices)", cv.WINDOW_NORMAL)
    cv.namedWindow("OP (slices)", cv.WINDOW_NORMAL)
    cv.moveWindow("CT (slices)", 200, 50)
    cv.moveWindow("GT (slices)", 200, 450)
    cv.moveWindow("OP (slices)", 200, 850)
    cv.resizeWindow("CT (slices)", 1800, 350)
    cv.resizeWindow("GT (slices)", 1800, 350)
    cv.resizeWindow("OP (slices)", 1800, 350)

    cv.namedWindow("CT (largest)", cv.WINDOW_NORMAL)
    cv.namedWindow("GT (largest)", cv.WINDOW_NORMAL)
    cv.namedWindow("OP (largest)", cv.WINDOW_NORMAL)
    cv.moveWindow("CT (largest)", 2100, 50)
    cv.moveWindow("GT (largest)", 2100, 450)
    cv.moveWindow("OP (largest)", 2100, 850)
    cv.resizeWindow("CT (largest)", 350, 350)
    cv.resizeWindow("GT (largest)", 350, 350)
    cv.resizeWindow("OP (largest)", 350, 350)


def destroy_windows():
    cv.destroyWindow("CT (slices)")
    cv.destroyWindow("GT (slices)")
    cv.destroyWindow("OP (slices)")
    cv.destroyWindow("CT (largest)")
    cv.destroyWindow("GT (largest)")
    cv.destroyWindow("OP (largest)")


def draw_data_target_output(data, target, output, x):
    try:
        data_np = data[x]

        target_np = target[x]
        target_np = np.argmax(target_np, axis=0)

        output_np = output[x]
        output_np = np.argmax(output_np, axis=0)

        # draw_pt = draw_volume_and_segmentation(data_pt_np, None, axis=2)
        draw_ct = draw_volume_and_segmentation(data_np[0], output_np, axis=2, normalize=True)
        draw_gt = draw_volume_and_segmentation(data_np[0], target_np, axis=2)
        # cv.imshow("DRAW_PT", draw_pt)

        cv.imshow("CT (slices)", draw_ct)
        cv.imshow("GT (slices)", draw_gt)
    except Exception as e:
        print(f"I barfed: {e}")


def draw_data_target_output_largest(data, target, output, x):
    try:
        target_np = np.squeeze(target[x], axis=0)
        data_pt_np = data[x, 0, :, :, :]
        data_ct_np = data[x, 1, :, :, :]

        output_np = output[x]
        output_np = np.argmax(output_np, axis=0)

        # draw_pt = draw_volume_and_segmentation(data_pt_np, None, axis=2)
        draw_ct = draw_volume_and_segmentation_largest(data_ct_np, output_np, axis=2, normalize=True)
        draw_gt = draw_volume_and_segmentation_largest(data_pt_np, target_np, axis=2)
        draw_op = draw_volume_and_segmentation_largest(data_pt_np, output_np, axis=2)
        # cv.imshow("DRAW_PT", draw_pt)

        cv.imshow("CT (largest)", draw_ct)
        cv.imshow("GT (largest)", draw_gt)
        cv.imshow("OP (largest)", draw_op)
    except:
        pass
