import math
import os

import cv2
import numpy as np


# copied from base.common_util import get_file_path_list
def get_file_path_list(path, ext=None):
    assert os.path.exists(path), 'path not exist {}'.format(path)
    assert ext is not None, 'ext is None'
    if os.path.isfile(path):
        return [path]
    file_path_list = []
    for root, _, files in os.walk(path):
        for file in files:
            try:
                if file.rsplit('.')[-1].lower() in ext:
                    file_path_list.append(os.path.join(root, file))
            except Exception as e:
                pass
    return file_path_list


def get_iou(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def rotate(image, angle):
    h, w = image.shape[:2]
    rad = np.pi / 180 * angle
    center = ((w - 1) // 2, (h - 1) // 2)
    h_new = int(math.ceil(abs(h * math.cos(rad)) + abs(w * math.sin(rad))))
    w_new = int(math.ceil(abs(w * math.cos(rad)) + abs(h * math.sin(rad))))
    rotate_matrix = cv2.getRotationMatrix2D(center, angle=angle, scale=1)
    rotate_matrix[0, 2] += w // 2 + (w_new // 2 - w)
    rotate_matrix[1, 2] += h // 2 + (h_new // 2 - h)
    transform_matrix = rotate_matrix
    dst_size = (w_new, h_new)
    return cv2.warpAffine(image, transform_matrix, dst_size, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def get_ocr_data_from_pdf(pdf_path):
    assert os.path.getsize(pdf_path) != 0, 'error'
    import fitz
    doc = fitz.Document(pdf_path)
    page = doc.load_page(0)
    text_block_list = page.getText(option='dict')['blocks']
    words_list = page.getText(option='words')
    return text_block_list, words_list


def sort_point(points1, points2):
    x10 = np.min(points1[:, 0])
    x11 = np.max(points1[:, 0])
    y10 = np.min(points1[:, 1])
    y11 = np.max(points1[:, 1])

    x20 = np.min(points2[:, 0])
    x21 = np.max(points2[:, 0])
    y20 = np.min(points2[:, 1])
    y21 = np.max(points2[:, 1])

    if y11 <= y20:
        return 1
    elif y21 <= y10:
        return -1
    y_com = min(y11, y21) - max(y10, y20)
    min_h = min(y11 - y10, y21 - y20)
    if y_com / min_h >= 0.5:
        # the same line
        if (x10 + x11) < (x20 + x21):
            return 1
        elif (x10 + x11) > (x20 + x21):
            return -1
        else:
            return 0
    else:
        if x11 <= x20 or x21 <= x10:
            if (y10 + y11) < (y20 + y21):
                return 1
            elif (y10 + y11) > (y20 + y21):
                return -1
            else:
                return 0
        x_com = min(x20, x21) - max(x10, x20)
        min_w = min(x11 - x10, x21 - x20)
        if x_com / min_w >= 0.2 and y_com / min_h >= 0.2:
            # the same line
            if (x10 + x11) < (x20 + x21):
                return 1
            elif (x10 + x11) > (x20 + x21):
                return -1
            else:
                return 0
        else:
            # not the same line
            if (y10 + y11) < (y20 + y21):
                return 1
            elif (y10 + y11) > (y20 + y21):
                return -1
            else:
                return 0
