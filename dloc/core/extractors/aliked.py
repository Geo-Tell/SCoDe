#!/usr/bin/env python
"""
@File    :   aliked.py
@Time    :   2024/01/01 00:00:00
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import sys
from pathlib import Path

import torch
import cv2
import numpy as np

# Add ALIKED path if it exists in third_party
aliked_path = Path(__file__).parent / '../../../third_party/ALIKED'
if aliked_path.exists():
    sys.path.append(str(aliked_path))
    try:
        from nets.aliked import ALIKED as ALIKEDModel
        ALIKED_AVAILABLE = True
    except ImportError:
        ALIKED_AVAILABLE = False
else:
    ALIKED_AVAILABLE = False

from ..utils.base_model import BaseModel


class ALIKED(BaseModel):
    """ALIKED: A Lighter Keypoint and Descriptor Extraction Network.
    
    ALIKED: A Lighter Keypoint and Descriptor Extraction Network
    Zhao, Xiaoming and Wu, Xingming and Miao, Jinyu and Chen, Weihai and 
    Chen, Peter C. Y. and Li, Zhengguo
    arXiv preprint arXiv:2304.03608, 2023.
    """

    default_conf = {
        'model_name': 'aliked-n16',
        'max_keypoints': 2048,
        'detection_threshold': 0.2,
        'top_k': -1,
        'nms_radius': 2,
        'load_pretrained': True,
    }
    required_inputs = ['image']

    def _init(self, conf, model_path):
        self.conf = {**self.default_conf, **conf}
        
        if not ALIKED_AVAILABLE:
            raise ImportError("ALIKED is not available. Please install ALIKED in third_party/ALIKED")
        
        # Initialize ALIKED model with proper parameters
        model_name = self.conf['model_name']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create model using the exact API from ALIKED source
        self.model = ALIKEDModel(
            model_name=model_name,
            device=device,
            top_k=self.conf['top_k'],
            scores_th=self.conf['detection_threshold'],
            n_limit=self.conf['max_keypoints'],
            load_pretrained=self.conf['load_pretrained']
        )
        
        # Model is already loaded and moved to device in ALIKED constructor
        self.model.eval()

    def _forward(self, data):
        # Get the image tensor
        image_tensor = data['image']
        
        # ALIKED expects RGB image in CHW format
        if image_tensor.shape[1] == 1:  # Grayscale to RGB
            image_tensor = image_tensor.repeat(1, 3, 1, 1)
        
        # ALIKED forward method returns a dictionary
        with torch.no_grad():
            pred = self.model(image_tensor)
        
        # Extract keypoints, descriptors, and scores
        keypoints = pred['keypoints']  # [B, N, 2]
        descriptors = pred['descriptors']  # [B, N, D]
        scores = pred['scores']  # [B, N]
        
        return {
            'keypoints': [keypoints[0]],
            'scores': [scores[0]],
            'descriptors': [descriptors[0].T],  # Transpose to match expected format [D, N]
        }
