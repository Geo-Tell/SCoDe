import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models


def box_tlbr_to_xyxy(loc, tlbr, max_h, max_w):
    # cxywh: [N, 4]
    t, l, b, r = tlbr.unbind(-1)
    x, y = loc.unbind(-1)
    t, b = t * max_h, b * max_h
    l, r = l * max_w, r * max_w

    x1 = (x - l).clamp(min=0.0, max=max_w)
    y1 = (y - t).clamp(min=0.0, max=max_h)
    x2 = (x + r).clamp(min=0.0, max=max_w)
    y2 = (y + b).clamp(min=0.0, max=max_h)
    b = [x1, y1, x2, y2]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxywh(xyxy, max_h, max_w):
    # cxywh: [N, 4]
    x1, y1, x2, y2 = xyxy.unbind(-1)
    x1 = x1.clamp(min=0.0, max=max_w)
    x2 = x2.clamp(min=0.0, max=max_w)
    y1 = y1.clamp(min=0.0, max=max_h)
    y2 = y2.clamp(min=0.0, max=max_h)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    b = [cx, cy, w, h]
    return torch.stack(b, dim=-1)


class ResnetEncoder(nn.Module):
    def __init__(self, cfg):
        super(ResnetEncoder, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.cfg = cfg
        self.last_layer = cfg.BACKBONE.LAST_LAYER

        resnets = {
            # 18: models.resnet18,
            # 34: models.resnet34,
            50: models.resnet50,
            # 101: models.resnet101,
            # 152: models.resnet152
        }
        # pdb.set_trace()
        encoder = resnets[cfg.BACKBONE.NUM_LAYERS](True)
        self.encoder = encoder

        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        if cfg.BACKBONE.LAYER == 'layer4':
            self.layer4 = encoder.layer4
        del encoder

        if cfg.BACKBONE.NUM_LAYERS > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        x = input_image.permute(0, 3, 1, 2).contiguous()
        # Normalize the input colorspace
        if self.cfg.NORM_INPUT:
            x = (x - 0.45) / 0.225

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.cfg.BACKBONE.LAYER == 'layer3':
            x = self.layer3(x)
        elif self.cfg.BACKBONE.LAYER == 'layer4':
            x = self.layer3(x)
            x = self.layer4(x)

        return x
