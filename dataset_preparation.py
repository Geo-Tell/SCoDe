import os
import math
import h5py
import torch
import random
import argparse
import warnings
import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from struct import unpack
from scipy import ndimage
from torch.nn import functional as F

from src.utils.utils import MVSCamera, compute_valid_map
from valid_utils import Pose

# from backup.ccoe.utils.utils import MVSCamera, compute_valid_map

warnings.filterwarnings("ignore")
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def read_corr(file_path):
    """Read the match correspondence file.
    Args:
        file_path: file path.
    Returns:
        matches: list of match data, each consists of two image indices and Nx15 match matrix, of
        which each line consists of two 2x3 transformations, geometric distance and two feature
        indices.
    """
    matches = []
    with open(file_path, 'rb') as fin:
        while True:
            rin = fin.read(24)
            if len(rin) == 0:
                # EOF
                break
            idx0, idx1, num = unpack('L' * 3, rin)
            bytes_theta = num * 60
            corr = np.fromstring(fin.read(bytes_theta), dtype=np.float32).reshape(-1, 15)
            matches.append([idx0, idx1, corr])
    return matches


def read_mask(file_path, size=32):
    """Read the mask file.
    Args:
        file_path: file path.
        size: mask size.
    Returns:
        mask_dict: mask data in dictionary, indexed by hashed pair index.
    """
    mask_dict = {}
    size = size * size * 2
    record_size = 8 + size

    with open(file_path, 'rb') as fin:
        data = fin.read()
    for i in range(0, len(data), record_size):
        decoded = unpack('2i' + '?' * size, data[i: i + record_size])
        mask = np.array(decoded[2:])
        mask_dict[hash_int_pair(decoded[0], decoded[1])] = mask
    return mask_dict


def hash_int_pair(ind1, ind2):
    """Hash an int pair.
    Args:
        ind1: int1.
        ind2: int2.
    Returns:
        hash_index: the hash index.
    """
    assert ind1 <= ind2
    return ind1 * 2147483647 + ind2


def read_cams(cam_path):
    """
    Args:
        cam_path: Path to cameras.txt.
    Returns:
        cam_dict: A dictionary indexed by image index and composed of (K, t, R, dist, img_size).
        K - 2x3, t - 3x1, R - 3x3, dist - 1x3, img_size - 1x2.
    """
    cam_data = [i.split(' ') for i in read_list(cam_path)]

    cam_dict = {}
    for i in cam_data:
        i = [float(j) for j in i if j is not '']
        K = np.array([(i[1], i[5], i[3]),
                      (0, i[2], i[4]), (0, 0, 1)])
        t = np.array([(i[6],), (i[7],), (i[8],)])
        R = np.array([(i[9], i[10], i[11]),
                      (i[12], i[13], i[14]),
                      (i[15], i[16], i[17])])
        dist = np.array([i[18], i[19], i[20]])
        img_size = np.array([i[21], i[22]])
        cam_dict[i[0]] = (K, t, R, dist, img_size)
    return cam_dict


def read_list(list_path):
    """Read list."""
    if list_path is None or not os.path.exists(list_path):
        print('Not exist', list_path)
        exit(-1)
    content = open(list_path).read().splitlines()
    return content


def main(args):
    scene_path = osp.join(args.base_path, 'scene_info')
    full_pair_path = osp.join(args.base_path, 'overlap_valid_pairs')

    if not osp.exists(full_pair_path):
        Path(full_pair_path).mkdir(exist_ok=True, parents=True)

    for root, dirs, files in os.walk(scene_path):
        if args.group >= 0:
            files = files[args.group * args.group_number:args.group * args.group_number + args.group_number]
        for i, scene_name in enumerate(files):
            if scene_name[:4] in args.valid_scene:
                print(scene_name + ' {}/{}'.format(i + 1, len(files)))

                scene_info = osp.join(root, scene_name)
                scene = np.load(scene_info, allow_pickle=True)
                image_paths = scene['image_paths']
                depth_paths = scene['depth_paths']
                intrinsics = scene['intrinsics']
                poses = scene['poses']
                overlap_ids = np.array(np.where(scene['overlap_matrix'] > 0)).T

                pairs = overlap(args, image_paths, depth_paths, intrinsics, poses, overlap_ids)

                with open(osp.join(full_pair_path, scene_name.split('.')[0] + '.txt'), 'w') as f:
                    f.writelines(pairs)


def overlap(args, image_paths, depth_paths, intrinsics, poses, overlap_ids):
    pairs = []
    overlap_ids = [k for k in overlap_ids]
    random.shuffle(overlap_ids)
    for im1_id, im2_id in tqdm(overlap_ids[:args.num_per_scene]):
        depth1 = torch.Tensor(h5py.File(osp.join(args.base_path, depth_paths[im1_id]))['depth'][:]).cuda()
        depth2 = torch.Tensor(h5py.File(osp.join(args.base_path, depth_paths[im2_id]))['depth'][:]).cuda()
        shape1 = depth1.shape
        shape2 = depth2.shape
        w = max(shape1[0], shape2[0])
        h = max(shape1[1], shape2[1])
        depth1 = F.pad(depth1, (0, h - shape1[1], 0, w - shape1[0]))
        depth2 = F.pad(depth2, (0, h - shape2[1], 0, w - shape2[0]))
        shape = depth1.shape

        depth12 = torch.stack((depth1, depth2), dim=0).unsqueeze(3)
        poses12 = torch.Tensor(np.stack((poses[im1_id], poses[im2_id]), axis=0)).cuda()
        intrinsics12 = torch.Tensor(np.stack((intrinsics[im1_id], intrinsics[im2_id]), axis=0)).cuda()
        bbox1 = get_bbox(intrinsics12, poses12, depth12, shape)
        if bbox1 == -1:
            continue

        depth21 = torch.stack((depth2, depth1), dim=0).unsqueeze(3)
        poses21 = torch.Tensor(np.stack((poses[im2_id], poses[im1_id]), axis=0)).cuda()
        intrinsics21 = torch.Tensor(np.stack((intrinsics[im2_id], intrinsics[im1_id]), axis=0)).cuda()
        bbox2 = get_bbox(intrinsics21, poses21, depth21, shape)
        if bbox2 == -1:
            continue

        bbox1 = ','.join([str(x) for x in bbox1])
        bbox2 = ','.join([str(x) for x in bbox2])
        pairs.append(' '.join(
            [image_paths[im1_id], depth_paths[im1_id], ','.join(intrinsics[im1_id].flatten().astype(np.str)),
             ','.join(poses[im1_id].flatten().astype(np.str)), bbox1,
             image_paths[im2_id], depth_paths[im2_id], ','.join(intrinsics[im2_id].flatten().astype(np.str)),
             ','.join(poses[im2_id].flatten().astype(np.str)), bbox2]) + '\n')

    return pairs


def get_bbox(intrinsics, poses, depth, shape):
    cameras = MVSCamera(intrinsics, poses, shape)
    valid_map = compute_valid_map(cameras, depth)
    valid_idx = np.where(valid_map.squeeze().cpu().numpy())

    if len(valid_idx[0]) <= 1:
        return -1

    x1 = min(valid_idx[0])
    x2 = max(valid_idx[0])
    y1 = min(valid_idx[1])
    y2 = max(valid_idx[1])

    bbox = (y1, x1, y2, x2)

    return bbox


def generate_pairs(args):
    full_pair_path = osp.join(args.base_path, 'overlap_pairs')
    image_path = osp.join(args.base_path, 'Undistorted_SfM')

    train_file_path = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_from_depth_train_V2.txt')
    valid_file_path = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_from_depth_valid_V2.txt')

    scene_image = {}
    for root, dirs, files in os.walk(image_path):
        root_name = Path(root).name
        for scene in dirs:
            for r, d, images in os.walk(osp.join(root, scene, 'images')):
                if len(images) <= 0:
                    break
                scene_image[scene] = [osp.join(root_name, scene, 'images', i) for i in images]

    scenes_train = list(scene_image.keys())
    scenes_valid = args.valid_scene
    for x in scenes_valid:
        scenes_train.remove(x)
    for root, dirs, files in os.walk(full_pair_path):
        for i, pair_name in enumerate(files):
            print(pair_name + ' {}/{}'.format(i + 1, len(files)))
            this_scene = Path(pair_name).stem

            if this_scene in scenes_train:
                scenes_c = scenes_train.copy()
                num = args.num_per_scene
                f_path = train_file_path
            else:
                scenes_c = scenes_valid.copy()
                num = args.valid_num_per_scene
                f_path = valid_file_path
            scenes_c.remove(this_scene)
            pair_info = osp.join(root, pair_name)
            with open(pair_info, 'r') as f:
                pairs = f.readlines()
            with open(f_path, 'a+') as f:
                for pair in pairs[:num]:
                    neg = random.choice(scene_image[random.choice(scenes_c)])
                    pair = ' '.join([pair[:-1], neg])
                    f.write(pair + '\n')


def generate_valid_pairs(args):
    full_pair_path = osp.join(args.base_path, 'overlap_valid_pairs')
    image_path = osp.join(args.base_path, 'Undistorted_SfM')

    file_path = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_Test.txt')

    scene_image = {}
    for root, dirs, files in os.walk(image_path):
        root_name = Path(root).name
        for scene in dirs:
            for r, d, images in os.walk(osp.join(root, scene, 'images')):
                if len(images) <= 0:
                    break
                scene_image[scene] = [osp.join(root_name, scene, 'images', i) for i in images]

    scenes_valid = args.valid_scene
    for root, dirs, files in os.walk(full_pair_path):
        for i, pair_name in enumerate(files):
            print(pair_name + ' {}/{}'.format(i + 1, len(files)))
            this_scene = Path(pair_name).stem
            scenes_c = scenes_valid.copy()
            f_path = file_path
            scenes_c.remove(this_scene)
            pair_info = osp.join(root, pair_name)
            with open(pair_info, 'r') as f:
                pairs = f.readlines()
            with open(f_path, 'a+') as f:
                for pair in pairs:
                    neg = random.choice(scene_image[random.choice(scenes_c)])
                    pair = ' '.join([pair[:-1], neg])
                    f.write(pair + '\n')


def generate_pairs_from_off(args):
    image_path = osp.join(args.base_path, 'Undistorted_SfM')

    train_file_path = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_from_scale_train_V2.txt')
    valid_file_path = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_from_scale_valid_V2.txt')

    train_file_path_off = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_from_scale_train.txt')
    valid_file_path_off = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_from_scale_valid.txt')

    scene_image = {}
    for root, dirs, files in os.walk(image_path):
        root_name = Path(root).name
        for scene in dirs:
            for r, d, images in os.walk(osp.join(root, scene, 'images')):
                if len(images) <= 0:
                    break
                scene_image[scene] = [osp.join(root_name, scene, 'images', i) for i in images]

    scenes_train = list(scene_image.keys())
    scenes_valid = args.valid_scene
    for x in scenes_valid:
        scenes_train.remove(x)

    with open(train_file_path_off, 'r') as f:
        train_pairs = [x.split() for x in f.readlines()]
    with open(valid_file_path_off, 'r') as f:
        valid_pairs = [x.split() for x in f.readlines()]
    train_pairs_V2 = []
    for pair in train_pairs:
        p_scene = pair[0].split('/')[1]
        scenes_c = scenes_train.copy()
        scenes_c.remove(p_scene)
        neg = random.choice(scene_image[random.choice(scenes_c)])
        pair.append(neg)
        train_pairs_V2.append(' '.join(pair) + '\n')
    with open(train_file_path, 'w') as f:
        f.writelines(train_pairs_V2)
    valid_pairs_V2 = []
    for pair in valid_pairs:
        p_scene = pair[0].split('/')[1]
        scenes_c = scenes_valid.copy()
        scenes_c.remove(p_scene)
        neg = random.choice(scene_image[random.choice(scenes_c)])
        pair.append(neg)
        valid_pairs_V2.append(' '.join(pair) + '\n')
    with open(valid_file_path, 'w') as f:
        f.writelines(valid_pairs_V2)


def generate_scale_pairs(args):
    valid_file_path_13 = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_Val_Scale_13.txt')
    valid_file_path_35 = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_Val_Scale_35.txt')
    valid_file_path_57 = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_Val_Scale_57.txt')
    valid_file_path_79 = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_Val_Scale_79.txt')
    valid_file_path_90 = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_Val_Scale_90.txt')
    # valid_file_path_off = osp.join(Path(args.base_path).parent, 'assets', 'megadepth_validation_scale.txt')
    valid_file_path_off = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_Test.txt')

    scenes_valid = args.valid_scene
    scene_flag_13 = dict.fromkeys(scenes_valid, 0)
    scene_flag_35 = dict.fromkeys(scenes_valid, 0)
    scene_flag_57 = dict.fromkeys(scenes_valid, 0)
    scene_flag_79 = dict.fromkeys(scenes_valid, 0)
    scene_flag_90 = dict.fromkeys(scenes_valid, 0)
    thres = 300

    with open(valid_file_path_off, 'r') as f:
        valid_pairs = [x.split() for x in f.readlines()]

    valid_pairs_scale_13 = []
    valid_pairs_scale_35 = []
    valid_pairs_scale_57 = []
    valid_pairs_scale_79 = []
    valid_pairs_scale_90 = []
    for pair in tqdm(valid_pairs):
        p_scene = pair[0].split('/')[1]
        if p_scene in scenes_valid:
            path1 = pair[0]
            intri1 = ' '.join(pair[2].split(','))
            extri1 = np.array(pair[3].split(','), dtype=float).reshape(4, 4)
            box1 = np.array(pair[4].split(','), dtype=int)
            box1_s = ' '.join(box1.astype(np.str))
            path2 = pair[5]
            intri2 = ' '.join(pair[7].split(','))
            extri2 = np.array(pair[8].split(','), dtype=float).reshape(4, 4)
            box2 = np.array(pair[9].split(','), dtype=int)
            box2_s = ' '.join(box2.astype(np.str))
            T = get_T_0to1(extri2, extri1)
            T = ' '.join(T.flatten().astype(np.str))
            p = ' '.join([path1, path2, intri1, intri2, T, box1_s, box2_s])
            scale = cal_scale(box1, box2)
            if scale < 3 and scene_flag_13[p_scene] < thres:
                scene_flag_13[p_scene] += 1
                valid_pairs_scale_13.append(p + '\n')
            elif 3 <= scale < 5 and scene_flag_35[p_scene] < thres:
                scene_flag_35[p_scene] += 1
                valid_pairs_scale_35.append(p + '\n')
            elif 5 <= scale < 7 and scene_flag_57[p_scene] < thres:
                scene_flag_57[p_scene] += 1
                valid_pairs_scale_57.append(p + '\n')
            elif 7 <= scale < 9 and scene_flag_79[p_scene] < thres:
                scene_flag_79[p_scene] += 1
                valid_pairs_scale_79.append(p + '\n')
            elif scale >= 9 and scene_flag_90[p_scene] < thres:
                scene_flag_90[p_scene] += 1
                valid_pairs_scale_90.append(p + '\n')

    with open(valid_file_path_13, 'w') as f:
        f.writelines(valid_pairs_scale_13)
    with open(valid_file_path_35, 'w') as f:
        f.writelines(valid_pairs_scale_35)
    with open(valid_file_path_57, 'w') as f:
        f.writelines(valid_pairs_scale_57)
    with open(valid_file_path_79, 'w') as f:
        f.writelines(valid_pairs_scale_79)
    with open(valid_file_path_90, 'w') as f:
        f.writelines(valid_pairs_scale_90)


def get_T_0to1(T1, T2):
    T1 = torch.tensor(T1)
    T2 = torch.tensor(T2)

    T_w2cam_1 = Pose.from_4x4mat(T1)
    T_w2cam_2 = Pose.from_4x4mat(T2)

    T_0to1 = T_w2cam_1 @ T_w2cam_2.inv()
    T_0to1 = np.array(torch.cat((torch.cat((T_0to1.R, T_0to1.t.unsqueeze(-1)), -1), torch.tensor([[0, 0, 0, 1]]))))

    return T_0to1


def cal_scale(box1, box2):
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]

    scale = (w1 + h1) / (w2 + h2)
    if scale < 1:
        scale = 1 / scale

    return scale


def statistic_scale_pairs(args):
    valid_file_path = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_Val_Scales.txt')
    # valid_file_path_off = osp.join(Path(args.base_path).parent, 'assets', 'megadepth_validation_scale.txt')
    valid_file_path_off = osp.join(Path(args.base_path).parent, 'assets', 'MegaDepth_Test.txt')

    scenes_valid = args.valid_scene
    scene_flag = dict.fromkeys(scenes_valid, 0)
    for k in scene_flag:
        scene_flag[k] = [0, 0, 0, 0, 0]
    thres = 200

    with open(valid_file_path_off, 'r') as f:
        valid_pairs = [x.split() for x in f.readlines()]

    valid_pairs_scale = []
    for pair in tqdm(valid_pairs):
        p_scene = pair[0].split('/')[1]
        if p_scene in scenes_valid:
            path1 = pair[0]
            intri1 = ' '.join(pair[2].split(','))
            extri1 = np.array(pair[3].split(','), dtype=float).reshape(4, 4)
            box1 = np.array(pair[4].split(','), dtype=int)
            box1_s = ' '.join(box1.astype(np.str))
            path2 = pair[5]
            intri2 = ' '.join(pair[7].split(','))
            extri2 = np.array(pair[8].split(','), dtype=float).reshape(4, 4)
            box2 = np.array(pair[9].split(','), dtype=int)
            box2_s = ' '.join(box2.astype(np.str))
            T = get_T_0to1(extri2, extri1)
            T = ' '.join(T.flatten().astype(np.str))
            scale = cal_scale(box1, box2)
            scale_level = math.floor(scale) - 1 if scale < 6 else 4
            p = ' '.join([path1, path2, intri1, intri2, T, box1_s, box2_s, str(scale)])
            if scene_flag[p_scene][scale_level] < thres:
                scene_flag[p_scene][scale_level] += 1
                valid_pairs_scale.append(p + '\n')

    with open(valid_file_path, 'w') as f:
        f.writelines(valid_pairs_scale)


def generate_val_pairs():
    from src.config.default import get_cfg_defaults
    from src.model import CCOE
    from src.datasets import build_dataloader
    from src.utils.validation import evaluate
    from src.utils.utils import get_logger

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/ccoe_config.py')
    model = CCOE(cfg.CCOE).eval().to(device)
    # model.load_state_dict(torch.load('weights/scode_1024_pt3.pth', map_location='cpu'))
    # model.load_state_dict(torch.load('weights/scode.pth', map_location='cpu'))
    model.load_state_dict(torch.load('weights/scode_NoMSA.pth', map_location='cpu'))
    model.eval()

    logger = get_logger('outputs/temp/temp.log')

    validation_dataset = build_dataloader(cfg.DATASET.VAL, cfg.DATASET.DATA_ROOT)
    validation_dataset.build_dataset()
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=4,
        num_workers=0,
        shuffle=False
    )

    evaluate(
        model,
        validation_dataloader,
        logger,
        iou_thrs=np.arange(0.5, 0.96, 0.05),
        oiou=cfg.DATASET.VAL.OIOU,
        save_results=True,
        output_dir='outputs/eval'
    )


def rewrite_scannet_test():
    path = "/Data/lhj/Datasets/scannet/scannet_test_pairs_with_gt.txt"
    with open(path, 'r') as f:
        pairs = [x.split() for x in f.readlines()]
    npairs = []
    for pair in tqdm(pairs):
        path1 = pair[0].split('/')
        path2 = pair[1].split('/')

        scene = path1[1]
        id1 = str(int(path1[-1][6:12]))
        id2 = str(int(path2[-1][6:12]))

        path1 = '/'.join(['scannet_test_1500', scene, 'color', id1 + '.jpg'])
        path2 = '/'.join(['scannet_test_1500', scene, 'color', id2 + '.jpg'])

        npairs.append(' '.join([path1, path2] + pair[2:]) + '\n')
    with open('/Data/lhj/Datasets/scannet/scannet_test_pairs_with_gt_modified.txt', 'w') as f:
        f.writelines(npairs)


def get_gl3d_pairs():
    path = '/Data/lhj/Datasets/gl3d/data'

    pairs = []
    for rt, dirs, files in os.walk(path):
        for scene in tqdm(dirs):
            root = osp.join(path, scene)
            corr_path = osp.join(root, 'geolabel', 'corr.bin')
            match_records = read_corr(corr_path)
            for match in match_records:
                cidx0 = match[0]
                cidx1 = match[1]
                basename0 = str(cidx0).zfill(8)
                basename1 = str(cidx1).zfill(8)
                img_path0 = os.path.join(root, 'undist_images', basename0 + '.jpg')
                img_path1 = os.path.join(root, 'undist_images', basename1 + '.jpg')
                pairs.append(' '.join([img_path0, img_path1]) + '\n')

        with open('/Data/lhj/Datasets/gl3d/gl3d_pairs.txt', 'w') as f:
            f.writelines(pairs)


def generate_gl3d_gt(args):
    path = '/Data/lhj/Datasets/gl3d/data'
    length = 1000

    valid_file_path_12 = osp.join(Path(args.base_path).parent, 'assets', 'GL3D_Val_Scale_12.txt')
    valid_file_path_23 = osp.join(Path(args.base_path).parent, 'assets', 'GL3D_Val_Scale_23.txt')
    valid_file_path_34 = osp.join(Path(args.base_path).parent, 'assets', 'GL3D_Val_Scale_34.txt')
    valid_file_path_45 = osp.join(Path(args.base_path).parent, 'assets', 'GL3D_Val_Scale_45.txt')
    valid_pairs_scale_12 = []
    valid_pairs_scale_23 = []
    valid_pairs_scale_34 = []
    valid_pairs_scale_45 = []

    pairs = []
    for rt, dirs, files in os.walk(path):
        for scene in tqdm(dirs):
            root = osp.join(path, scene)
            corr_path = osp.join(root, 'geolabel', 'corr.bin')
            match_records = read_corr(corr_path)
            cam_path = os.path.join(root, 'geolabel', 'cameras.txt')
            cam_dict = read_cams(cam_path)
            mask_path = osp.join(root, 'geolabel', 'mask.bin')
            mask_dict = read_mask(mask_path)
            for match in match_records:
                cidx0 = match[0]
                cidx1 = match[1]
                basename0 = str(cidx0).zfill(8)
                basename1 = str(cidx1).zfill(8)
                img_path0 = os.path.join(root, 'undist_images', basename0 + '.jpg')
                img_path1 = os.path.join(root, 'undist_images', basename1 + '.jpg')
                pairs.append(' '.join([img_path0, img_path1]) + '\n')

                cam0, cam1 = cam_dict[cidx0], cam_dict[cidx1]
                K0, K1 = cam0[0], cam1[0]
                t0, t1 = cam0[1], cam1[1]
                R0, R1 = cam0[2], cam1[2]
                extri0 = np.vstack((np.hstack((R0, t0)), np.array([0, 0, 0, 1])))
                extri1 = np.vstack((np.hstack((R1, t1)), np.array([0, 0, 0, 1])))
                # T = get_T_0to1(extri1, extri0)
                T = get_T_0to1(extri0, extri1)
                K0 = ' '.join(K0.flatten().astype(np.str))
                K1 = ' '.join(K1.flatten().astype(np.str))
                T = ' '.join(T.flatten().astype(np.str))

                mask = mask_dict.get(hash_int_pair(cidx0, cidx1))
                if mask is None:
                    continue
                size = 32
                mask0 = ndimage.binary_fill_holes(np.reshape(mask[:size * size], (size, size)))
                mask1 = ndimage.binary_fill_holes(np.reshape(mask[size * size:], (size, size)))
                bbox0 = bounding(mask0, length / size)
                bbox1 = bounding(mask1, length / size)
                bbox0_s = ' '.join(bbox0.astype(np.str))
                bbox1_s = ' '.join(bbox1.astype(np.str))
                p = ' '.join([img_path0, img_path1, K0, K1, T, bbox0_s, bbox1_s])
                scale = cal_scale(bbox0, bbox1)

                if scale < 2:
                    valid_pairs_scale_12.append(p + '\n')
                elif 2 <= scale < 3:
                    valid_pairs_scale_23.append(p + '\n')
                elif 3 <= scale < 4:
                    valid_pairs_scale_34.append(p + '\n')
                elif scale >= 4:
                    valid_pairs_scale_45.append(p + '\n')

        with open(valid_file_path_12, 'w') as f:
            f.writelines(valid_pairs_scale_12)
        with open(valid_file_path_23, 'w') as f:
            f.writelines(valid_pairs_scale_23)
        with open(valid_file_path_34, 'w') as f:
            f.writelines(valid_pairs_scale_34)
        with open(valid_file_path_45, 'w') as f:
            f.writelines(valid_pairs_scale_45)


def bounding(mask, ratio):
    mask_idx = np.where(mask)

    if len(mask_idx[0]) <= 1:
        return -1

    x1 = int(min(mask_idx[0]) * ratio)
    x2 = int(max(mask_idx[0]) * ratio)
    y1 = int(min(mask_idx[1]) * ratio)
    y2 = int(max(mask_idx[1]) * ratio)

    bbox = np.array([y1, x1, y2, x2])

    return bbox


def reorg_gl3d(args):
    dirs = os.listdir('/Data/lhj/Datasets/gl3d/data')
    scenes = random.sample(dirs, 10)

    valid_file_path_12 = osp.join(Path(args.base_path).parent, 'assets', 'GL3D_Val_Scale_12.txt')
    valid_file_path_23 = osp.join(Path(args.base_path).parent, 'assets', 'GL3D_Val_Scale_23.txt')
    valid_file_path_34 = osp.join(Path(args.base_path).parent, 'assets', 'GL3D_Val_Scale_34.txt')
    valid_file_path_45 = osp.join(Path(args.base_path).parent, 'assets', 'GL3D_Val_Scale_45.txt')

    pair_12 = smaller_pairs(valid_file_path_12, scenes)
    pair_23 = smaller_pairs(valid_file_path_23, scenes)
    pair_34 = smaller_pairs(valid_file_path_34, scenes)
    pair_45 = smaller_pairs(valid_file_path_45, scenes)

    gl3d_path_12 = 'dataset/gl3d/GL3D_Val_Scale_12.txt'
    gl3d_path_23 = 'dataset/gl3d/GL3D_Val_Scale_23.txt'
    gl3d_path_34 = 'dataset/gl3d/GL3D_Val_Scale_34.txt'
    gl3d_path_45 = 'dataset/gl3d/GL3D_Val_Scale_45.txt'

    with open(gl3d_path_12, 'w') as f:
        f.writelines(pair_12)
    with open(gl3d_path_23, 'w') as f:
        f.writelines(pair_23)
    with open(gl3d_path_34, 'w') as f:
        f.writelines(pair_34)
    with open(gl3d_path_45, 'w') as f:
        f.writelines(pair_45)


def smaller_pairs(in_path, scenes):
    # scene_flag = dict.fromkeys(scenes, 0)
    # thres = 100
    #
    # with open(in_path, 'r') as f:
    #     pairs = [x.split() for x in f.readlines()]
    #
    # random.shuffle(pairs)
    #
    # n_p = []
    # for pair in pairs:
    #     scene = pair[0][-51:-27]
    #     if (scene in scenes) and (scene_flag[scene] < thres):
    #         scene_flag[scene] += 1
    #         n_p.append(' '.join(pair) + '\n')

    with open(in_path, 'r') as f:
        pairs = [x for x in f.readlines()]

    random.shuffle(pairs)
    n_p = pairs[:1000]

    return n_p


def generate_scale_sets():
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate training overlap pairs using processed MegaDepth images and depth maps',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--base_path',
                        type=str,
                        default='dataset/megadepth/MegaDepth',
                        help='base path of MegaDepth dataset')
    parser.add_argument('--num_per_scene',
                        type=int,
                        default=5000,
                        help='pairs chosen per scene for train')
    parser.add_argument('--valid_num_per_scene',
                        type=int,
                        default=500,
                        help='pairs chosen per scene for valid')
    parser.add_argument('--valid_scene',
                        type=int,
                        default=['0024', '0021', '0019', '1589', '0025', '0022', '0008', '0063', '0032', '0015'],
                        help='scenes for valid')
    parser.add_argument('--group',
                        type=int,
                        default=-1,
                        help='gpu group')
    parser.add_argument('--group_number',
                        type=int,
                        default=5,
                        help='gpu group number')
    args = parser.parse_args()

    # main(args)
    # generate_pairs(args)
    # generate_pairs_from_off(args)
    generate_val_pairs()
    # rewrite_scannet_test()
    # get_gl3d_pairs()
    # generate_scale_pairs(args)
    # generate_valid_pairs(args)
    # generate_gl3d_gt(args)
    # reorg_gl3d(args)
    # statistic_scale_pairs(args)
