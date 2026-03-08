import torch
import logging
import cv2 as cv
import numpy as np
import torch.nn.functional as F


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s]' +
        '%(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')


def to_GB(memory):
    return round(memory / 1024 ** 3, 1)


def get_gpu_memory(name='', id=0):
    t = torch.cuda.get_device_properties(id).total_memory
    # c = torch.cuda.memory_reserved(id)
    a = torch.cuda.memory_allocated(id)
    # f = c-a  # free inside cache
    return to_GB(a), to_GB(t)
    # print('Free GPU memory : {}/{}'.format(f, t))
    # logger.info('{} GPU memory : {}/{} GB'.format(name, to_GB(a), to_GB(t)))


def process_resize(w, h, resize):
    assert len(resize) > 0 and len(resize) <= 2
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.0).float()[None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv.imread(str(path))
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


def visualize_overlap(image1, bbox1, image2, bbox2, output):
    left = cv.rectangle(image1, tuple(bbox1[0:2]), tuple(bbox1[2:]),
                        (255, 0, 0), 2)
    right = cv.rectangle(image2, tuple(bbox2[0:2]), tuple(bbox2[2:]),
                         (0, 0, 255), 2)
    viz = cv.hconcat([left, right])
    cv.imwrite(output, viz)


def visualize_overlap_gt(image1,
                         bbox1,
                         gt1,
                         image2,
                         bbox2,
                         gt2,
                         output,
                         save=True):
    left = cv.rectangle(image1, tuple(bbox1[0:2]), tuple(bbox1[2:]),
                        (255, 0, 0), 2)
    right = cv.rectangle(image2, tuple(bbox2[0:2]), tuple(bbox2[2:]),
                         (0, 0, 255), 2)
    left = cv.rectangle(left, tuple(gt1[0:2]), tuple(gt1[2:]), (0, 255, 0), 2)
    right = cv.rectangle(right, tuple(gt2[0:2]), tuple(gt2[2:]), (0, 255, 0),
                         2)
    if save:
        viz = cv.hconcat([left, right])
        cv.imwrite(output, viz)
    return left, right


def visualize_centerness_overlap_gt(image1, bbox1, gt1, center1, image2, bbox2,
                                    gt2, center2, output):
    left, right = visualize_overlap_gt(image1, bbox1, gt1, image2, bbox2, gt2,
                                       output, False)
    # from IPython import embed;embed()
    center1 = (center1 - center1.min()) / center1.max()
    center2 = (center2 - center2.min()) / center2.max()
    center1 = cv.resize(center1.astype('float32'), image1.shape[:-1]) * 255
    center2 = cv.resize(center2.astype('float32'), image2.shape[:-1]) * 255

    center1 = cv.applyColorMap(center1.astype(np.uint8), cv.COLORMAP_JET)
    center2 = cv.applyColorMap(center2.astype(np.uint8), cv.COLORMAP_JET)

    left = cv.addWeighted(left, 1.0, center1.astype('float32'), 0.4, 0)
    right = cv.addWeighted(right, 1.0, center2.astype('float32'), 0.4, 0)

    viz = cv.hconcat([left, right])
    cv.imwrite(output, viz)


def visualization_heatmap():
    pass


def loss_info(infos, writer, iter):
    str_info = ''
    for k, v in infos.items():
        if 'loss' in k:
            str_info += '{}:{:.5f}, '.format(k, v)
            writer.add_scalar('Loss/{}'.format(k), v, iter)
        if 'iou' in k:
            str_info += '{}:{:.5f}, '.format(k, v)
            writer.add_scalar('iou/{}'.format(k), v, iter)
    return str_info


def from_homogeneous(points):
    return points[:, :, :, :-1] / points[:, :, :, -1:]


def to_homogeneous(points):
    dims = list(points.shape)
    dims[3] = 1
    ones = torch.ones(tuple(dims), dtype=points.dtype, device=points.device)
    return torch.cat((points, ones), 3)


def to_vector(points):
    return points.squeeze(-1)


def from_vector(points):
    return points.unsqueeze(-1)


def to_bchw(bhwc):
    return bhwc.permute(0, 3, 1, 2)


def to_bhwc(bchw):
    return bchw.permute(0, 2, 3, 1)


class MVSCamera:
    def __init__(self, K, E, shape):
        """
        Creates commonly used variables from intrinsics, extrinsics

        Args:
            K: Bx3x3 intrinsic matrix
            E: Bx4x4 extrinsic matrix
            shape: (H, W) image size
        """
        self.H, self.W = shape
        self.K = K.view(-1, 1, 1, 3, 3)
        self.E = E.view(-1, 1, 1, 4, 4)
        self.R = self.E[:, :, :, :3, :3]
        self.t = self.E[:, :, :, :3, 3:]
        self.Ki = self.inverse_intrinsic(self.K)
        self.Rt = self.R.transpose(-1, -2)
        self.grid_K = self.generate_grid_intrinsics()

    def inverse_intrinsic(self, K):
        Ki = K.clone()
        fx = K[..., 0:1, 0:1]
        fy = K[..., 1:2, 1:2]
        Ki[..., 0:1, :] /= fx
        Ki[..., 1:2, :] /= fy
        Ki[..., 0:1, 0:1] /= fx
        Ki[..., 1:2, 1:2] /= fy
        Ki[..., :2, -1] *= -1
        return Ki

    def generate_grid_intrinsics(self):
        dev = self.K.device
        grid_K = torch.zeros(2, 3, device=dev)
        grid_K[0, 0] = 2 / float(self.W - 1)
        grid_K[0, 2] = -float(self.W) / float(self.W - 1)
        grid_K[1, 1] = 2 / float(self.H - 1)
        grid_K[1, 2] = -float(self.H) / float(self.H - 1)
        return grid_K.view(1, 1, 1, 2, 3)

    ###############################
    # inverse projection related
    # all projection related takes
    # NxHxWx... format shapes and outputs NxHxWx... format shapes
    def pixel_points(self, offset=0.5):
        """
        Given width and height, creates a mesh grid, and returns homogeneous
        coordinates
        of image in a 3 x W*H Tensor

        Arguments:
            width {Number} -- Number representing width of pixel grid image
            height {Number} -- Number representing height of pixel grid image

        Returns:
            torch.Tensor -- 1x2xHxW, oriented in x, y order
        """
        dev = self.K.device
        W = self.W
        H = self.H
        O = offset
        x_coords = torch.linspace(O, W - 1 + O, W, device=dev)
        y_coords = torch.linspace(O, H - 1 + O, H, device=dev)

        # HxW grids
        y_grid_coords, x_grid_coords = torch.meshgrid([y_coords, x_coords])

        # HxWx2 grids => 1xHxWx2 grids
        return torch.stack([x_grid_coords.contiguous(), y_grid_coords.contiguous()], 2).unsqueeze(0)

    def camera_rays(self, num_views=1):
        x = from_vector(to_homogeneous(self.pixel_points()))
        Ki = self.Ki[:num_views]
        return to_vector(Ki @ x)

    def back_project(self, depth_maps, num_views=1):
        """
        Given depth map, back project its depth to obtain world coordinates

        Args:
            depth_maps: NxHxWx1 depths
        Returns:
            NxHxWx3 points in world coordinates
        """
        Rt = self.Rt[:num_views]
        t = self.t[:num_views]

        # 1xHxWx3 * NxHxWx1
        r = self.camera_rays(1)
        p = from_vector(r * depth_maps)

        return to_vector(Rt @ (p - t))

    def project(self, world_p, num_views=1):
        p = from_vector(world_p)
        K = self.K[:num_views]
        R = self.R[:num_views]
        t = self.t[:num_views]

        return to_vector(from_homogeneous(K @ (R @ p + t)))

    def normalize(self, points):
        # NxHxWx3x1
        projected = to_homogeneous(from_homogeneous(points))
        return self.grid_K @ projected


def compute_warps(cameras, depth_maps):
    world_p = cameras.back_project(depth_maps[:1], 1)
    projected = cameras.project(world_p, None)
    h_projected = to_homogeneous(projected)
    # NxHxWx2
    warped = to_vector(cameras.normalize(from_vector(h_projected)))
    return warped


def compute_valid_map(cameras: MVSCamera, depths):
    w = compute_warps(cameras, depths)
    valid_p = to_bhwc(F.grid_sample(to_bchw(depths), w, align_corners=False, mode='nearest')) > 0
    valids = (valid_p[:1] & valid_p[1:])

    return valids


def draw_bbox(img, box, score=None, gt_box=None, tag=True):
    img_b = img.copy()
    img_b[:box[1], :, :] = 255
    img_b[box[3]:, :, :] = 255
    img_b[:, :box[0], :] = 255
    img_b[:, box[2]:, :] = 255

    if tag:
        if gt_box is not None:
            cv.rectangle(img_b, (gt_box[2] - 215, gt_box[3] - 36), gt_box[2:], (71, 99, 255), -1)
        cv.rectangle(img_b, box[:2], (box[0] + 290, box[1] + 37), (255, 144, 30), -1)

    img = cv.addWeighted(img, 0.4, img_b, 0.6, 0)

    if gt_box is not None:
        cv.rectangle(img, gt_box[:2], gt_box[2:], (71, 99, 255), 2)
        if tag:
            cv.putText(
                img=img,
                text='Ground Truth',
                org=(gt_box[2] - 210, gt_box[3] - 8),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv.LINE_AA)

    cv.rectangle(img, box[:2], box[2:], (255, 144, 30), 2)
    # cv.rectangle(img, box[:2], box[2:], (71, 99, 255), 2)
    if tag:
        text = 'Our SCoDe' if score is None else 'Confidence:{:.3f}'.format(float(score))
        cv.putText(
            img=img,
            text=text,
            org=(box[0] + 5, box[1] + 30),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv.LINE_AA)

    return img


def visualize(img1: np.ndarray, img2: np.ndarray,
              box1: tuple, box2: tuple, score=None,
              gt_box1=None, gt_box2=None,
              tag=True):
    shape1 = img1.shape
    shape2 = img2.shape

    if shape1[0] == shape2[0]:
        img1 = draw_bbox(img1, box1, score, gt_box1, tag=tag)
        img2 = draw_bbox(img2, box2, score, gt_box2, tag=tag)
        img = cv.hconcat([img1, img2])
    elif shape1[0] > shape2[0]:
        r = shape2[0] / shape1[0]
        w = int(r * shape1[1])
        img1 = cv.resize(img1, (w, shape2[0]))
        box1 = tuple([int(x * r) for x in box1])
        img1 = draw_bbox(img1, box1, score, gt_box1, tag=tag)
        img2 = draw_bbox(img2, box2, score, gt_box2, tag=tag)
        img = cv.hconcat([img1, img2])
    else:
        r = shape1[0] / shape2[0]
        w = int(r * shape2[1])
        img2 = cv.resize(img2, (w, shape1[0]))
        box2 = tuple([int(x * r) for x in box2])
        img1 = draw_bbox(img1, box1, score, gt_box1, tag=tag)
        img2 = draw_bbox(img2, box2, score, gt_box2, tag=tag)
        img = cv.hconcat([img1, img2])

    return img
