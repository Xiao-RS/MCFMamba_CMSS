import torch.nn as nn

from models import get_model
from engine.utils import load_cpt
from ._loss import get_supvcomploss


class Adjust(nn.Module):
    def __init__(self, modeler_cfg, num_channels, num_classes, img_size):
        super().__init__()
        self.model = get_model(
            model_name=modeler_cfg.MODEL,
            num_channels=num_channels, num_classes=num_classes, img_size=img_size,
            **modeler_cfg.MODEL_SPECS[0]
        )
        if modeler_cfg.MODEL_PRE_PATH is not None:
            pre_ckpt = load_cpt(modeler_cfg.MODEL_PRE_PATH)["model"]
            del pre_ckpt['sem_seg_head.output.weight']
            self.model.load_state_dict(pre_ckpt, strict=False)
            print("Successfully load pre_ckpt")

        self.comploss_func = get_supvcomploss(**modeler_cfg.LOSS_SPECS[0])

    def forward(self, image, **kwargs):
        logits = self.model(image)

        losses_dict = self.comploss_func(logits, kwargs["target"])

        if self.training:
            return losses_dict
        else:
            return logits, losses_dict

    def get_learnable_parameters(self):
        return [v for k, v in self.model.named_parameters()]