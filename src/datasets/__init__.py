from .megadepth_pairs import MegaDepthPairsDataset


def build_dataloader(cfg, dataset_path):
    if cfg.DATA_SOURCE == 'megadepth_pairs':
        dataset = MegaDepthPairsDataset(
            pairs_list_path=cfg.LIST_PATH,
            base_path=dataset_path,
            pairs_per_scene=cfg.PAIRS_LENGTH,
            train=cfg.TRAIN,
            image_size=cfg.IMAGE_SIZE,
        )
        return dataset
    else:
        raise ValueError(f'DATASET {cfg.DATA_SOURCE} not supported.')
