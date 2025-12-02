import torch.nn as nn
from ._loss import Lmse

from models import get_model
from engine.utils import load_cpt


class SelfSupvLoss(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.losses_func = Lmse()
        self.losses_weights = {
            'self_supv_loss': 1,
        }
        self.channel_m1 = num_channels['m1']

    def forward(self, image, masked_out):
        mask1 = image['mask1'].repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
        mask2 = image['mask2'].repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
        pred_out_m1 = masked_out[:, :self.channel_m1] * mask1
        target_m1 = image['m1'] * mask1
        pred_out_m2 = masked_out[:, self.channel_m1:] * mask2
        target_m2 = image['m2'] * mask2
        selfsupv_m1_loss = self.losses_func(pred_out_m1, target_m1)
        selfsupv_m2_loss = self.losses_func(pred_out_m2, target_m2)

        losses_dict = {
            'self_supv_loss': selfsupv_m1_loss + selfsupv_m2_loss,
        }

        return losses_dict


class MSS(nn.Module):
    def __init__(self, modeler_cfg, num_channels, num_classes, img_size):
        super().__init__()
        self.model = get_model(
            model_name=modeler_cfg.MODEL,
            num_channels=num_channels, num_classes=sum(num_channels.values()), img_size=img_size,
            **modeler_cfg.MODEL_SPECS[0]
        )
        if modeler_cfg.MODEL_PRE_PATH is not None:
            pre_ckpt = load_cpt(modeler_cfg.MODEL_PRE_PATH)["model"]
            self.model.load_state_dict(pre_ckpt, strict=False)
            print("Successfully load pre_ckpt")

        self.comploss_func = SelfSupvLoss(num_channels=num_channels)

    def forward(self, image, **kwargs):
        masked_out = self.model(image, masked=True)

        losses_dict = self.comploss_func(
            image, masked_out
        )

        if self.training:
            # train
            return losses_dict

    def get_learnable_parameters(self):
        return [v for k, v in self.model.named_parameters()]