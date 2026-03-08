import torch
import argparse
import numpy as np
import os.path as osp
from pathlib import Path

# from src.model import OETR
from src.model import CCOE
from src.utils.utils import get_logger
from src.datasets import build_dataloader
from src.utils.validation import evaluate
from src.config.default import get_cfg_defaults

torch.set_grad_enabled(False)


def main(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_cfg_defaults()
    cfg.merge_from_file(opt.config_path)
    # model = OETR(cfg.OETR).eval().to(device)
    model = CCOE(cfg.CCOE).eval().to(device)
    model.load_state_dict(torch.load(opt.checkpoint, map_location='cpu'))

    logger = get_logger(
        osp.join(opt.save_path, '{}.log'.format('_'.join([Path(opt.checkpoint).parent.stem,
                                                          Path(opt.checkpoint).stem,
                                                          Path(cfg.DATASET.VAL.LIST_PATH).stem]))))

    validation_dataset = build_dataloader(cfg.DATASET.VAL, cfg.DATASET.DATA_ROOT)
    validation_dataset.build_dataset()
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False
    )

    evaluate(
        model,
        validation_dataloader,
        logger,
        iou_thrs=np.arange(0.5, 0.96, 0.05),
        oiou=cfg.DATASET.VAL.OIOU,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validate model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--save_path',
                        type=str,
                        default='outputs/validate_logs',
                        help='Path to the directory that contains the images')
    parser.add_argument('--checkpoint',
                        type=str,
                        # default='OUTPUT/SCoDe/pl128000_is480_ep40/model_epoch_18.pth',
                        # default='OUTPUT/SCoDe/pl128000_is480_ep30/model_epoch_29.pth',
                        default='weights/oetr/oetr_mf_epoch24_2x4_best.pth',
                        # default='weights/oetr/oetr_mf_epoch30_2x4_cyclecenter.pth',
                        help='Path to the checkpoints of model')
    parser.add_argument('--config_path',
                        type=str,
                        # default='configs/ccoe_config.py',
                        default='configs/baseline/oetr_config.py',
                        help='Path to the configuration of model')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='batch_size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='num_workers')
    opt = parser.parse_args()

    Path(opt.save_path).mkdir(exist_ok=True, parents=True)
    main(opt)
