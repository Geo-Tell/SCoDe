import os
import cv2
import argparse
import numpy as np


def resize_dataset(img, image_size, depth=False, neg=False):
    """resize image with different tyle."""
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape
    # resize w*h

    if neg:
        return cv2.resize(img, tuple(image_size))
    if w > h:
        if depth:
            img1 = cv2.resize(img, (int(image_size[0] / h * w), image_size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            img1 = cv2.resize(img, (int(image_size[0] / h * w), image_size[0]))
        resize_ratio = (int(image_size[0] / h * w) / w, image_size[0] / h)
    else:
        if depth:
            img1 = cv2.resize(img, (image_size[0], int(image_size[0] * h / w)), interpolation=cv2.INTER_NEAREST)
        else:
            img1 = cv2.resize(img, (image_size[0], int(image_size[0] * h / w)))
        resize_ratio = (image_size[0] / w, int(image_size[0] * h / w) / h)
    return img1, resize_ratio


def get_boxes(points):
    """calculate boundary box of point cloud."""
    box = np.array(
        [points[0].min(), points[1].min(), points[0].max(), points[1].max()])
    return box


def get_maskes(points, h, w):
    """calculate the mask mat of point cloud, output binary mask."""
    points = points.astype(int)
    mask = np.zeros((h, w))
    mask[points[1], points[0]] = 1
    return mask


def numpy_overlap_box(K1, depth1, pose1, bbox1, ratio1, K2, depth2, pose2,
                      bbox2, ratio2):
    """calculate numpy array co-visible bounding box."""
    mask1 = np.where(depth1 > 0)
    u1, v1 = mask1[1], mask1[0]
    Z1 = depth1[v1, u1]

    # COLMAP convention
    x1 = (u1 + bbox1[1] + 0.5) / ratio1[1]
    y1 = (v1 + bbox1[0] + 0.5) / ratio1[0]
    X1 = (x1 - K1[0, 2]) * (Z1 / K1[0, 0])
    Y1 = (y1 - K1[1, 2]) * (Z1 / K1[1, 1])
    # Homogeneous coordinates
    XYZ1_hom = np.concatenate(
        [
            X1.reshape(1, -1),
            Y1.reshape(1, -1),
            Z1.reshape(1, -1),
            np.ones_like(Z1.reshape(1, -1)),
        ],
        axis=0,
    )
    # Warp points to camera2
    XYZ2_hom = pose2 @ np.linalg.inv(pose1) @ XYZ1_hom
    XYZ2 = XYZ2_hom[:-1, :] / XYZ2_hom[-1, :].reshape(1, -1)

    uv2_hom = K2 @ XYZ2
    uv2 = uv2_hom[:-1, :] / uv2_hom[-1, :].reshape(1, -1)
    h, w = depth2.shape
    u2 = uv2[0, :] * ratio2[1] - bbox2[1] - 0.5
    v2 = uv2[1, :] * ratio2[0] - bbox2[0] - 0.5
    uv2 = np.concatenate([u2.reshape(1, -1), v2.reshape(1, -1)], axis=0)
    i = uv2[0, :].astype(int)
    j = uv2[1, :].astype(int)

    valid_corners = np.logical_and(np.logical_and(i >= 0, j >= 0),
                                   np.logical_and(i < h, j < w))

    valid_uv1 = np.stack((u1[valid_corners], v1[valid_corners])).astype(int)
    valid_uv2 = uv2[:, valid_corners].astype(int)

    # depth validation
    Z2 = depth2[valid_uv2[1], valid_uv2[0]]
    inlier_mask = np.absolute(XYZ2[2, valid_corners] - Z2) < 0.5

    valid_uv1 = valid_uv1[:, inlier_mask]
    valid_uv2 = valid_uv2[:, inlier_mask]

    if valid_uv1.shape[1] == 0 or valid_uv2.shape[1] == 0:
        return (
            np.array([0] * 4),
            np.zeros((h, w)),
            np.array([0] * 4),
            np.zeros((h, w)),
            False,
        )
    # box with x1, y1, x2, y2
    box1 = get_boxes(valid_uv1)
    box2 = get_boxes(valid_uv2)
    # mask
    mask1 = get_maskes(valid_uv1, h, w)
    mask2 = get_maskes(valid_uv2, h, w)
    return box1, mask1, box2, mask2, True


def crop(image1, image2, central_match, image_size):
    """crop patch from the central match and protect patch, not outside the
    boundary."""
    bbox1_i = max(int(central_match[0]) - image_size[0] // 2, 0)
    if bbox1_i + image_size[0] >= image1.shape[0]:
        bbox1_i = image1.shape[0] - image_size[0]
    bbox1_j = max(int(central_match[1]) - image_size[1] // 2, 0)
    if bbox1_j + image_size[1] >= image1.shape[1]:
        bbox1_j = image1.shape[1] - image_size[1]

    bbox2_i = max(int(central_match[2]) - image_size[1] // 2, 0)
    if bbox2_i + image_size[1] >= image2.shape[0]:
        bbox2_i = image2.shape[0] - image_size[1]
    bbox2_j = max(int(central_match[3]) - image_size[1] // 2, 0)
    if bbox2_j + image_size[1] >= image2.shape[1]:
        bbox2_j = image2.shape[1] - image_size[1]

    return (
        image1[bbox1_i:bbox1_i + image_size[0], bbox1_j:bbox1_j + image_size[1], ],
        np.array([bbox1_i, bbox1_j]),
        image2[bbox2_i:bbox2_i + image_size[0], bbox2_j:bbox2_j + image_size[1], ],
        np.array([bbox2_i, bbox2_j]),
    )


def crop_with_validation(image1, image2, central_match, image_size):
    """改进的裁剪函数，包含更好的边界检查"""
    crop_h, crop_w = image_size[0], image_size[1] if len(image_size) > 1 else image_size[0]
    
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    central_match[0] = np.clip(central_match[0], crop_h//2, h1 - crop_h//2)
    central_match[1] = np.clip(central_match[1], crop_w//2, w1 - crop_w//2)
    central_match[2] = np.clip(central_match[2], crop_h//2, h2 - crop_h//2)
    central_match[3] = np.clip(central_match[3], crop_w//2, w2 - crop_w//2)
    
    # Image1 crop
    bbox1_i = int(central_match[0]) - crop_h // 2
    bbox1_j = int(central_match[1]) - crop_w // 2
    
    # Image2 crop
    bbox2_i = int(central_match[2]) - crop_h // 2
    bbox2_j = int(central_match[3]) - crop_w // 2
    
    bbox1_i = max(0, min(bbox1_i, h1 - crop_h))
    bbox1_j = max(0, min(bbox1_j, w1 - crop_w))
    bbox2_i = max(0, min(bbox2_i, h2 - crop_h))
    bbox2_j = max(0, min(bbox2_j, w2 - crop_w))

    return (
        image1[bbox1_i:bbox1_i + crop_h, bbox1_j:bbox1_j + crop_w],
        np.array([bbox1_i, bbox1_j]),
        image2[bbox2_i:bbox2_i + crop_h, bbox2_j:bbox2_j + crop_w],
        np.array([bbox2_i, bbox2_j]),
    )


def recover_pair(base_path,
                 image_size,
                 pair_metadata,
                 with_gray=False):
    """calculate image pairs information from metadata.

    Args:
        pair_metadata (dict): contains image path
    """
    image_path1 = os.path.join(base_path, pair_metadata['image_path1'])
    if with_gray:
        image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    else:
        image1 = cv2.imread(image_path1)

    image_path2 = os.path.join(base_path, pair_metadata['image_path2'])
    if with_gray:
        image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    else:
        image2 = cv2.imread(image_path2)

    image_path_n = os.path.join(base_path, pair_metadata['image_path_n'])
    if with_gray:
        image_n = cv2.imread(image_path_n, cv2.IMREAD_GRAYSCALE)
    else:
        image_n = cv2.imread(image_path_n)

    # Resize data
    image1, resize_ratio1 = resize_dataset(image1, image_size)
    image2, resize_ratio2 = resize_dataset(image2, image_size)
    image_n = resize_dataset(image_n, image_size, neg=True)

    central_match = pair_metadata['central_match'] * np.concatenate((resize_ratio1, resize_ratio2))

    # image1, cbbox1, image2, cbbox2 = crop(image1, image2, central_match, image_size)
    image1, cbbox1, image2, cbbox2 = crop_with_validation(image1, image2, central_match, image_size)

    bbox1 = resize_bbox(pair_metadata['overlap1'], resize_ratio1, cbbox1, image_size)
    bbox2 = resize_bbox(pair_metadata['overlap2'], resize_ratio2, cbbox2, image_size)

    return (
        image1,
        bbox1,
        resize_ratio1,
        image2,
        bbox2,
        resize_ratio2,
        image_n
    )


def resize_bbox(bbox, ratio, cbbox, image_size):
    bbox = bbox.copy().astype(np.float32)
    
    bbox[0] *= ratio[0]  # x1 * width_ratio
    bbox[2] *= ratio[0]  # x2 * width_ratio  
    bbox[1] *= ratio[1]  # y1 * height_ratio
    bbox[3] *= ratio[1]  # y2 * height_ratio

    bbox[0] -= cbbox[1]  # x1 - crop_x_offset
    bbox[2] -= cbbox[1]  # x2 - crop_x_offset
    bbox[1] -= cbbox[0]  # y1 - crop_y_offset
    bbox[3] -= cbbox[0]  # y2 - crop_y_offset

    bbox[0] = max(0, bbox[0])
    bbox[1] = max(0, bbox[1])
    bbox[2] = min(image_size[0], bbox[2])
    bbox[3] = min(image_size[1], bbox[3])
    
    min_size = 5
    if bbox[2] - bbox[0] < min_size:
        center_x = (bbox[0] + bbox[2]) / 2
        bbox[0] = max(0, center_x - min_size/2)
        bbox[2] = min(image_size[0], bbox[0] + min_size)
    
    if bbox[3] - bbox[1] < min_size:
        center_y = (bbox[1] + bbox[3]) / 2
        bbox[1] = max(0, center_y - min_size/2)
        bbox[3] = min(image_size[1], bbox[1] + min_size)

    return bbox.astype(np.int32)


# print image width and height
def statistics_image_pairs(pairs_list_path, dataset_path):
    with open(pairs_list_path, 'r') as f:
        for line in f.readlines():
            image_path1 = os.path.join(dataset_path, line.split()[0])
            image1 = cv2.imread(image_path1)
            image_path2 = os.path.join(dataset_path, line.split()[5])
            image2 = cv2.imread(image_path2)
            print(image1.shape, image2.shape)


# Main pipeline
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate megadepth image pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--pairs_list_path',
        type=str,
        default='assets/megadepth_validation.txt',
        help='Path to the list of scenes',
    )
    parser.add_argument('--dataset_path',
                        type=str,
                        default='',
                        help='path to the dataset')
    args = parser.parse_args()

    statistics_image_pairs(**args.__dict__)
