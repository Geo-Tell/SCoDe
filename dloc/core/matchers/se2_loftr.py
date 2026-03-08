#!/usr/bin/env python
"""
@File    :   se2_loftr.py
@Time    :   2024/01/01 00:00:00
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# 修正SE2-LoFTR路径计算
current_file = Path(__file__).resolve()
se2_loftr_path = current_file.parents[3] / 'third_party' / 'se2-loftr'
src_path = se2_loftr_path / 'src'
sys.path.insert(0, str(src_path))  # 添加src目录到路径

print(f"Looking for SE2-LoFTR at: {se2_loftr_path}")
print(f"Adding src path: {src_path}")

if se2_loftr_path.exists() and src_path.exists():
    try:
        # 从src.loftr导入，因为__init__.py在src/loftr/目录下
        from loftr import LoFTR as SE2LoFTRModel
        from loftr.utils.cvpr_ds_e2_config import default_cfg
        SE2_LOFTR_AVAILABLE = True
        print("SE2-LoFTR modules imported successfully")
    except ImportError as e:
        print(f"Failed to import SE2-LoFTR modules: {e}")
        SE2_LOFTR_AVAILABLE = False
else:
    print(f"SE2-LoFTR directory not found at: {se2_loftr_path}")
    SE2_LOFTR_AVAILABLE = False

from ..utils.base_model import BaseModel


class SE2LoFTR(BaseModel):
    """SE2-LoFTR Detector-Free Local Feature Matching."""

    default_conf = {
        'weights': 'se2_loftr/outdoor_ds.ckpt',
        'match_threshold': 0.2,
        'max_keypoints': 2048,
    }
    required_inputs = [
        'image0',
        'image1',
    ]

    def _init(self, conf, model_path):
        self.conf = {**self.default_conf, **conf}
        
        if not SE2_LOFTR_AVAILABLE:
            raise ImportError("SE2-LoFTR is not available. Please check the installation path.")
        
        try:
            # 使用默认配置
            config = default_cfg
            config['match_coarse']['thr'] = self.conf['match_threshold']
            
            # 初始化模型
            self.model = SE2LoFTRModel(config)
            
            # 加载权重
            weights_path = model_path / self.conf['weights']
            if weights_path.exists():
                print(f"Loading weights from: {weights_path}")
                checkpoint = torch.load(str(weights_path), map_location='cpu')
                
                # 处理不同的权重格式
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    # 移除'matcher.'前缀如果存在
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_key = key.replace('matcher.', '') if key.startswith('matcher.') else key
                        new_state_dict[new_key] = value
                    state_dict = new_state_dict
                else:
                    state_dict = checkpoint
                
                self.model.load_state_dict(state_dict, strict=False)
                print("Weights loaded successfully")
            else:
                print(f"Warning: Weights file not found at {weights_path}")
            
            self.model.eval()
            print("SE2-LoFTR model initialized successfully")
            
        except Exception as e:
            print(f"Error initializing SE2-LoFTR: {e}")
            raise

    def _forward(self, data):
        try:
            # SE2-LoFTR需要灰度图像
            image0 = data['image0']
            image1 = data['image1']
            
            # 转换为灰度图像
            if image0.shape[1] == 3:
                image0 = 0.299 * image0[:, 0:1] + 0.587 * image0[:, 1:2] + 0.114 * image0[:, 2:3]
            if image1.shape[1] == 3:
                image1 = 0.299 * image1[:, 0:1] + 0.587 * image1[:, 1:2] + 0.114 * image1[:, 2:3]
            
            # 确保两张图片尺寸一致 - 重要修复
            h0, w0 = image0.shape[2], image0.shape[3]
            h1, w1 = image1.shape[2], image1.shape[3]
            
            # 如果尺寸不一致，将两张图片都resize到较小的尺寸
            if h0 != h1 or w0 != w1:
                target_h = min(h0, h1)
                target_w = min(w0, w1)
                
                # 确保尺寸是8的倍数（SE2-LoFTR的要求）
                target_h = (target_h // 8) * 8
                target_w = (target_w // 8) * 8
                
                if target_h > 0 and target_w > 0:
                    image0 = torch.nn.functional.interpolate(image0, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    image1 = torch.nn.functional.interpolate(image1, size=(target_h, target_w), mode='bilinear', align_corners=False)
                else:
                    # 如果计算出的尺寸太小，使用默认尺寸
                    target_h, target_w = 480, 640
                    image0 = torch.nn.functional.interpolate(image0, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    image1 = torch.nn.functional.interpolate(image1, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            # 准备批次数据
            batch = {
                'image0': image0,
                'image1': image1
            }
            
            # 前向传播
            with torch.no_grad():
                self.model(batch)
            
            # 提取匹配结果
            if 'mkpts0_f' in batch and 'mkpts1_f' in batch and len(batch['mkpts0_f']) > 0:
                mkpts0 = batch['mkpts0_f']  # 精细匹配
                mkpts1 = batch['mkpts1_f']
                mconf = batch['mconf']
            elif 'mkpts0_c' in batch and 'mkpts1_c' in batch and len(batch['mkpts0_c']) > 0:
                mkpts0 = batch['mkpts0_c']  # 粗略匹配
                mkpts1 = batch['mkpts1_c']
                mconf = batch.get('mconf', torch.ones(len(mkpts0), device=mkpts0.device))
            else:
                # 没有匹配点
                mkpts0 = torch.empty(0, 2, device=image0.device)
                mkpts1 = torch.empty(0, 2, device=image0.device)
                mconf = torch.empty(0, device=image0.device)
            
            # 限制关键点数量
            if self.conf['max_keypoints'] > 0 and mkpts0.shape[0] > self.conf['max_keypoints']:
                _, top_indices = torch.topk(mconf, self.conf['max_keypoints'])
                mkpts0 = mkpts0[top_indices]
                mkpts1 = mkpts1[top_indices]
                mconf = mconf[top_indices]
            
            # 创建匹配索引
            matches = torch.arange(mkpts0.shape[0], device=mkpts0.device)
            
            return {
                'keypoints0': [mkpts0],
                'keypoints1': [mkpts1], 
                'matches0': [matches],
                'matching_scores0': [mconf],
            }
            
        except Exception as e:
            print(f"Error in SE2-LoFTR forward pass: {e}")
            # 返回空结果
            return {
                'keypoints0': [torch.empty(0, 2, device=data['image0'].device)],
                'keypoints1': [torch.empty(0, 2, device=data['image0'].device)],
                'matches0': [torch.empty(0, dtype=torch.long, device=data['image0'].device)],
                'matching_scores0': [torch.empty(0, device=data['image0'].device)],
            }
