from src.config.default import _CN as cfg

cfg.OUTPUT = 'SCoDe'
cfg.CCOE.CHECKPOINT = 'weights/scode.pth'

# Dataset
cfg.DATASET.DATA_ROOT = './dataset/megadepth/MegaDepth'

# Training
cfg.DATASET.TRAIN.DATA_SOURCE = 'megadepth_pairs'
cfg.DATASET.TRAIN.LIST_PATH = 'dataset/megadepth/assets/MegaDepth_from_scale_train_V2.txt'
cfg.DATASET.TRAIN.PAIRS_LENGTH = 128000
cfg.DATASET.TRAIN.IMAGE_SIZE = [1024, 1024]

# Validation
cfg.DATASET.VAL.DATA_SOURCE = 'megadepth_pairs'
cfg.DATASET.VAL.LIST_PATH = 'dataset/megadepth/assets/MegaDepth_from_scale_valid_V2.txt'
cfg.DATASET.VAL.PAIRS_LENGTH = None
cfg.DATASET.VAL.IMAGE_SIZE = [1024, 1024]


# CCOE BACKBONE
cfg.CCOE.BACKBONE.NUM_LAYERS = 50
cfg.CCOE.BACKBONE.STRIDE = 32
cfg.CCOE.BACKBONE.LAYER = 'layer3'  # options: ['layer3', 'layer4']
cfg.CCOE.BACKBONE.LAST_LAYER = 1024  # output last channel size

# CCOE TRANSFORMER
cfg.CCOE.CCA.FEAT_SIZE = cfg.DATASET.TRAIN.IMAGE_SIZE[0] // (2 ** 5)
cfg.CCOE.CCA.FEAT_CHAN = cfg.CCOE.BACKBONE.LAST_LAYER // 4
cfg.CCOE.CCA.DEPTH = [2, 2, 2, 2]
cfg.CCOE.CCA.NUM_HEADS = [8, 8, 8, 8]
cfg.CCOE.CCA.MSA_SIZES = [[3, 5, 7], [3, 5, 7], [3, 5, 7]]
cfg.CCOE.CCA.NON_OVERLAP_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]]

# CCOE LOSS
cfg.CCOE.LOSS.OIOU = False
cfg.CCOE.LOSS.CYCLE_OVERLAP = True
