import torch.nn as nn

from models import get_model
from engine.utils import load_cpt
from ._loss import get_supvcomploss


class Vanilla(nn.Module):
    def __init__(self, modeler_cfg, num_channels, num_classes, img_size):
        super().__init__()
        self.model = get_model(
            model_name=modeler_cfg.MODEL,
            num_channels=num_channels, num_classes=num_classes, img_size=img_size,
            **modeler_cfg.MODEL_SPECS[0]
        )
        if modeler_cfg.MODEL_PRE_PATH is not None:
            state_dict = load_cpt(modeler_cfg.MODEL_PRE_PATH)["model"]
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            print("Successfully load pre_ckpt")

        self.comploss_func = get_supvcomploss(**modeler_cfg.LOSS_SPECS[0])

    def forward(self, image, **kwargs):
        logits = self.model(image)

        losses_dict = self.comploss_func(logits, kwargs["target"])

        if self.training:
            # train
            return losses_dict
        else:
            # val
            return logits, losses_dict

    def get_learnable_parameters(self):
        return [v for k, v in self.model.named_parameters()]
