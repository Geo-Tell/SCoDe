from yacs.config import CfgNode as CN

_CN = CN()
_CN.OUTPUT = ''

# CCOE Pipeline
_CN.CCOE = CN()
_CN.CCOE.CHECKPOINT = None
_CN.CCOE.BACKBONE_TYPE = 'ResNet'
_CN.CCOE.MODEL = 'ccoe'
_CN.CCOE.NORM_INPUT = True

# CCOE BACKBONE
_CN.CCOE.BACKBONE = CN()
_CN.CCOE.BACKBONE.NUM_LAYERS = 50
_CN.CCOE.BACKBONE.STRIDE = 16
_CN.CCOE.BACKBONE.LAYER = 'layer3'  # options: ['layer4']
_CN.CCOE.BACKBONE.LAST_LAYER = 1024  # output last channel size

# CCOE NECK
_CN.CCOE.NECK = CN()
_CN.CCOE.NECK.MAX_SHAPE = (100, 100)  # max feature map shape, with image shape: max_shape*stride

# CCOE TRANSFORMER
_CN.CCOE.CCA = CN()
_CN.CCOE.CCA.FEAT_SIZE = 40
_CN.CCOE.CCA.FEAT_CHAN = 256
_CN.CCOE.CCA.DEPTH = [2, 2, 6, 2]
_CN.CCOE.CCA.NUM_HEADS = [8, 8, 8, 8]
_CN.CCOE.CCA.MSA_SIZES = [[3, 5, 9], [3, 5, 9], [3, 5, 9], [3, 5, 9]]
_CN.CCOE.CCA.NON_OVERLAP_SIZES = [[2, 4, 8], [2, 4, 8], [2, 4, 8], [2, 4, 8]]

# CCOE HEAD
_CN.CCOE.HEAD = CN()
_CN.CCOE.HEAD.D_MODEL = 256
_CN.CCOE.HEAD.NORM_REG_TARGETS = True

# CCOE LOSS
_CN.CCOE.LOSS = CN()
_CN.CCOE.LOSS.OIOU = False
_CN.CCOE.LOSS.CYCLE_OVERLAP = False
_CN.CCOE.LOSS.FOCAL_ALPHA = 0.25
_CN.CCOE.LOSS.FOCAL_GAMMA = 2.0
_CN.CCOE.LOSS.REG_WEIGHT = 1.0
_CN.CCOE.LOSS.CENTERNESS_WEIGHT = 1.0

# Dataset
_CN.DATASET = CN()
_CN.DATASET.DATA_ROOT = None

# Training
_CN.DATASET.TRAIN = CN()
_CN.DATASET.TRAIN.DATA_SOURCE = 'megadepth'
_CN.DATASET.TRAIN.LIST_PATH = 'assets/train_scenes.txt'
_CN.DATASET.TRAIN.PAIRS_LENGTH = None
_CN.DATASET.TRAIN.WITH_MASK = None
_CN.DATASET.TRAIN.TRAIN = True
_CN.DATASET.TRAIN.IMAGE_SIZE = [640, 640]
_CN.DATASET.TRAIN.SCALES = [[1200, 1200], [1200, 1200]]

# Validation
_CN.DATASET.VAL = CN()
_CN.DATASET.VAL.DATA_SOURCE = 'megadepth'
_CN.DATASET.VAL.LIST_PATH = 'assets/val_scenes.txt'
_CN.DATASET.VAL.PAIRS_LENGTH = None
_CN.DATASET.VAL.WITH_MASK = False
_CN.DATASET.VAL.OIOU = True
_CN.DATASET.VAL.TRAIN = False
_CN.DATASET.VAL.IMAGE_SIZE = [640, 640]
_CN.DATASET.VAL.SCALES = [[1200, 1200], [1200, 1200]]


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
