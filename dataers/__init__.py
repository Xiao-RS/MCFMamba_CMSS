import torch
import numpy as np
from torch.utils.data import DataLoader

from .get_datalist import get_datalist
from .get_transforms import get_transforms


def get_dataloaders(cfg_DATASET, pattern='train_val'):
    dataset_name = cfg_DATASET.TRAIN_VAL_SETS
    if dataset_name == "DFC2020":
        from .DFC2020 import DFC2020 as dataset
    else:
        raise NotImplementedError(dataset_name)

    datalist_root, trainval_rate = get_datalist(dataset_name, **cfg_DATASET.SPLIT_SPECS[0], )

    transform = get_transforms(
        augmentation=cfg_DATASET.AUGMENT_SPECS[0]
    )

    if pattern == "train_val":
        if trainval_rate == 1.0:
            train_set = dataset(
                pattern='train', datalist_root=datalist_root, transform=transform["train"], **cfg_DATASET.LOAD_SPECS[0]
            )
            if 'drop_last' in cfg_DATASET.LOAD_SPECS[0].keys():
                drop_last = True
            else:
                drop_last = False
            if (len(train_set) % cfg_DATASET.BATCH_SIZE) == 1:
                drop_last = True
            train_loader = DataLoader(
                train_set, batch_size=cfg_DATASET.BATCH_SIZE, num_workers=cfg_DATASET.NUM_WORKERS, shuffle=True,
                drop_last=drop_last, worker_init_fn=worker_init_fn,
            )
            return train_loader, None, train_set.labels_dict, train_set.n_channels, train_set.img_size

        train_set = dataset(
            pattern='train', datalist_root=datalist_root, transform=transform["train"], **cfg_DATASET.LOAD_SPECS[0]
        )
        val_set = dataset(
            pattern='val', datalist_root=datalist_root, transform=transform["val"], **cfg_DATASET.LOAD_SPECS[0]
        )
        train_loader = DataLoader(
            train_set, batch_size=cfg_DATASET.BATCH_SIZE, num_workers=cfg_DATASET.NUM_WORKERS, shuffle=True,
            drop_last=True if (len(train_set) % cfg_DATASET.BATCH_SIZE) == 1 else False, worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_set, batch_size=cfg_DATASET.BATCH_SIZE, num_workers=cfg_DATASET.NUM_WORKERS, shuffle=False,
            drop_last=True if (len(val_set) % cfg_DATASET.BATCH_SIZE) == 1 else False, worker_init_fn=worker_init_fn,
        )
        return train_loader, val_loader, train_set.labels_dict, train_set.n_channels, train_set.img_size

    elif pattern == "test":
        test_set = dataset(
            pattern='test', datalist_root=datalist_root, transform=transform["val"], **cfg_DATASET.LOAD_SPECS[0],
        )
        test_loader = DataLoader(
            test_set, batch_size=cfg_DATASET.BATCH_SIZE, num_workers=cfg_DATASET.NUM_WORKERS, shuffle=False,
        )
        return test_loader, test_set.labels_dict, test_set.n_channels, test_set.img_size

    else:
        raise NotImplementedError(pattern)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    torch.manual_seed(seed)

