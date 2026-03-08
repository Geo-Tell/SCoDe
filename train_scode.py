import os
import torch
import random
import argparse
import datetime
import warnings
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from src.model import build_detectors
from src.datasets import build_dataloader
from src.utils.validation import evaluate
from src.config.default import get_cfg_defaults
from src.utils.utils import get_logger, loss_info

warnings.filterwarnings("ignore")


def setup_seed(seed):
    """set random seed to protect the training results.

    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def group_para(model):
    special_layers = torch.nn.ModuleList(
        [model.module.cls_conv, model.module.cls_reg]
    )
    special_layers_params = list(map(id, special_layers.parameters()))
    base_params = filter(lambda p: id(p) not in special_layers_params, model.parameters())
    return base_params, special_layers.parameters()


def main(opt):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(opt.config_path)
    setup_seed(opt.seed)
    # output folder init
    timestamp = datetime.datetime.now().strftime('%m-%d-%H:%M')
    opt.save_path = Path(f'./OUTPUT/{cfg.OUTPUT}/' + timestamp)

    opt.save_path.mkdir(exist_ok=True, parents=True)
    # pytorch init
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=datetime.timedelta(seconds=30000)
    )
    device = (torch.device(f'cuda:{opt.local_rank}') if torch.cuda.is_available() else torch.device('cpu'))

    # build dataloader and detectors
    training_dataset = build_dataloader(cfg.DATASET.TRAIN, cfg.DATASET.DATA_ROOT)
    model = build_detectors(cfg.CCOE).to(device)

    if cfg.CCOE.CHECKPOINT:
        model.load_state_dict(torch.load(cfg.CCOE.CHECKPOINT, map_location='cpu'))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], find_unused_parameters=True)
    base_params, flow_params = group_para(model)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': opt.learning_rate},
        {'params': flow_params, 'lr': opt.learning_rate * 0.2}
    ])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 25], gamma=0.1)
    # sim_loss, iouloss, wh_loss, loc_loss, cycle_losss
    # loss_weight = np.array([0.5, 0.25, 0.125, 0.125, 0.125])
    loss_weight = [1, 2, 2.5, 2.5, 2]
    # loss_weight = np.array([1, 1, 1, 1, 1])
    # loss_weight = np.array([2, 2.5, 2.5, 2])

    if opt.local_rank == 0:
        writer = SummaryWriter(opt.save_path / 'logs')
        logger = get_logger(os.path.join(opt.save_path, ('{}.log'.format(timestamp))))
        logger.info(opt)
        logger.info(cfg)
        logger.info(model)

    if opt.validation:
        validation_dataset = build_dataloader(cfg.DATASET.VAL, cfg.DATASET.DATA_ROOT)
        validation_dataset.build_dataset()
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            shuffle=False,
        )

    # start training
    for epoch in range(opt.epoch):
        model.float().train()
        training_dataset.build_dataset()

        train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset)
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler,
        )

        for i, batch in enumerate(training_dataloader):
            data = model(batch)
            # losses = np.array([_value for _key, _value in data.items() if 'loss' in _key])
            losses = [_value for _key, _value in data.items() if 'loss' in _key]
            pairs = zip(losses, loss_weight)
            loss = sum(loss * weight for loss, weight in pairs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save info
            if opt.local_rank == 0 and i % 50 == 0:
                info = loss_info(data, writer, i + epoch * len(training_dataloader))

                logger.info(
                    'Epoch [{}][{}/{}], lr-base: {:E}, lr-cls: {:E}, loss: {:.5f}, {}'.format(
                        epoch,
                        i,
                        len(training_dataloader),
                        scheduler.get_last_lr()[0],
                        scheduler.get_last_lr()[1],
                        loss,
                        info,
                    ))

                writer.add_scalar('Loss/train', loss.item(), i + epoch * len(training_dataloader))

        if opt.local_rank == 0:
            print('---------saving weights----------')
            model_out_path = opt.save_path / 'model_epoch_{}.pth'.format(epoch)
            torch.save(model.module.state_dict(), model_out_path)
            print('-------------DONE!---------------')

        # validation results
        if opt.local_rank == 0 and opt.validation:
            model.eval()
            evaluate(
                model,
                validation_dataloader,
                logger,
                iou_thrs=np.arange(0.5, 0.96, 0.05),
                oiou=cfg.DATASET.VAL.OIOU
            )
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate megadepth image pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--config_path',
                        type=str,
                        default='configs/scode_config.py',
                        help='configs of trainning',
                        )
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='batch_size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='num_workers')
    parser.add_argument('--local_rank', '--local-rank', 
                        type=int,
                        default=0,
                        help='node rank for distributed training')
    parser.add_argument('--validation',
                        action='store_true',
                        help='Use validation recalls')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epoch',
                        type=int,
                        default=40,
                        help='Number of epoches')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed')
    args = parser.parse_args()

    main(args)
