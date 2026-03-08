#!/usr/bin/env python
"""
@File    :   landmark.py
@Time    :   2021/06/28 11:07:03
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import cv2
import numpy as np
import torch
from sklearn.preprocessing import normalize

from ..utils.base_model import BaseModel


class Landmark(BaseModel):
    default_conf = {
        'sift': False,
    }
    required_inputs = ['image']

    def _init(self, conf, model_path):
        # using sift keypoints
        self.conf = {**self.default_conf, **conf}

        self.with_sift = self.conf['sift']
        if self.with_sift:
            self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.conf['topk'])

    def _forward(self, data):
        if self.with_sift:
            # image = convert_tensor_to_numpy(data['image'])
            # kpts = self.sift.detect(image)
            # kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
            # coord = torch.from_numpy(kpts).float()

            image = convert_tensor_to_numpy(data['image'])
            kpts, descs = self.sift.detectAndCompute(image, None)  # 计算关键点和描述子

            # 提取关键点坐标
            kpts_array = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
            # 提取关键点的 score (response)
            scores_array = np.array([kp.response for kp in kpts])
            # L2正则化
            descs = normalize(descs, norm='l2', axis=1)

            coord = torch.from_numpy(kpts_array).float()
            scores = torch.from_numpy(scores_array).float()
            descs = torch.from_numpy(descs).float().T if descs is not None else None  # 转换为 Tensor

            return {
                'keypoints': [coord],
                'scores': [scores],
                'descriptors': [descs],
            }
        else:
            return

def convert_tensor_to_numpy(tensor):
    # 确保输入 tensor 是 4D (BCHW)
    if len(tensor.shape) != 4:
        raise ValueError("Input tensor must be of shape BCHW")

    # 从批量中提取第一张图像
    tensor = tensor[0]

    # 如果是三通道图像 (CHW)，转换为 (HWC)
    if tensor.shape[0] == 3:  # 如果是彩色图像
        tensor = tensor.permute(1, 2, 0).contiguous()

    # 转换为 NumPy 数组
    numpy_image = tensor.cpu().numpy()

    # 确保数据类型是 uint8，并将归一化的 [0, 1] 范围转换为 [0, 255]
    if numpy_image.dtype != np.uint8:
        numpy_image = (numpy_image * 255).astype(np.uint8)

    return numpy_image

