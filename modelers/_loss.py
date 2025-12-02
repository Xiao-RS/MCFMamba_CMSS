import torch
import torch.nn as nn
from torch.nn import functional as F
# from modelers.supv_loss.bce import Re_BCEWithLogitsLoss
from segmentation_models_pytorch.losses import DiceLoss

Loss_Matcher = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    # 'BCEWithLogits': Re_BCEWithLogitsLoss,
    'DiceLoss': DiceLoss,
}


class CompositeLoss(nn.Module):

    def __init__(self, losses_params, losses_weight=None):
        super().__init__()
        self.losses_weights = losses_weight
        self.losses_func = {
            loss_name: get_singleloss(loss_name, params_dict) for loss_name, params_dict in losses_params.items()
        }

    def forward(self, outputs, targets):
        losses_dict = {}
        for loss_name, weight in self.losses_weights.items():
            losses_dict[loss_name] = weight * self.losses_func[loss_name](outputs, targets.to(torch.long))

        return losses_dict


def get_singleloss(loss_name, params_dict):
    assert loss_name in Loss_Matcher.keys(), "loss function not defined"
    if params_dict is None:
        return Loss_Matcher[loss_name]()
    else:
        return Loss_Matcher[loss_name](**params_dict)


def get_supvcomploss(losses_params="default", losses_weight=None):
    if isinstance(losses_params, dict):
        if losses_weight is None:
            losses_weight = {k: 1 for k in losses_params.keys()}

        if list(losses_params.keys()).sort() != list(losses_weight.keys()).sort():
            raise ValueError('The supv_loss and weights must have the same name keys.')

        return CompositeLoss(losses_params, losses_weight)
    elif losses_params == "default":
        return CompositeLoss(
            {'CrossEntropyLoss': None},
            {'CrossEntropyLoss': 1.0}
        )
    else:
        raise TypeError('The loss description is formatted improperly. See the docs for details.')


class Lmse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats_s, feats_t, **kwargs):
        loss = 0
        for feature_s, feature_t in zip(feats_s, feats_t):
            loss += F.mse_loss(feature_s, feature_t)
        loss /= len(feats_s)
        return loss
