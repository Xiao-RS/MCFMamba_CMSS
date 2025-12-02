# data augmenttaion

from .alb_core.composition import Compose
from .alb_trans import Flip, RandomRotate90
aug_matcher = {
    'Flip': Flip,
    'RandomRotate90': RandomRotate90,
}


def get_transforms(augmentation=None):
    if isinstance(augmentation, dict):
        if "train_augs" in augmentation.keys():
            if "additional_targets" not in augmentation.keys():
                raise ValueError('Check if you want to transform additional images')
            train_augs_list = _get_augs_list(augmentation["train_augs"])
            train_p = augmentation["augs_probs"].get('train_p', 1.0)
            transform_train = Compose(train_augs_list, p=train_p, additional_targets=augmentation["additional_targets"])
        else:
            transform_train = None
        if "val_augs" in augmentation.keys():
            val_augs_list = _get_augs_list(augmentation["val_augs"])
            val_p = augmentation["augs_probs"].get('val_p', 1.0)
            transform_val = Compose(val_augs_list, p=val_p, additional_targets=augmentation["additional_targets"])
        else:
            transform_val = None
    else:
        raise NotImplementedError(augmentation)

    transform = {"train": transform_train, "val": transform_val}

    return transform


def _get_augs_list(aug_dict):
    aug_list = []
    if aug_dict is not None:
        for aug_obj, params in aug_dict.items():
            if params is None or isinstance(params, dict):
                assert aug_obj in aug_matcher.keys(), "augmentation not defined"
                aug_list.append(aug_matcher[aug_obj](**params))
            else:
                raise ValueError('{} is not a valid aug param (must be dict of args)'.format(params))
    return aug_list
