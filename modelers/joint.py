import torch.nn as nn

from models import get_model
from ._loss import get_supvcomploss


class Supv_CompLoss(nn.Module):

    def __init__(self, loss_specs, **kwargs):
        super().__init__()
        supv_comp_loss = get_supvcomploss(**loss_specs)
        self.losses_weights = {
            'supv_m1_segloss': kwargs.get('m1_weight', 1),
            'supv_m2_segloss': kwargs.get('m2_weight', 1),
            'supv_fus_segloss': kwargs.get('fus_weight', 1),
        }
        self.losses_func = {
            'supv_m1_segloss': supv_comp_loss,
            'supv_m2_segloss': supv_comp_loss,
            'supv_fus_segloss': supv_comp_loss,
        }

    def forward(self, logits_m1, logits_m2, logits_fus, target):
        supv_m1_segloss = sum(self.losses_func['supv_m1_segloss'](logits_m1, target).values())
        supv_m2_segloss = sum(self.losses_func['supv_m2_segloss'](logits_m2, target).values())
        supv_fus_segloss = sum(self.losses_func['supv_fus_segloss'](logits_fus, target).values())

        losses_dict = {
            'supv_m1_segloss': supv_m1_segloss * self.losses_weights['supv_m1_segloss'],
            'supv_m2_segloss': supv_m2_segloss * self.losses_weights['supv_m2_segloss'],
            'supv_fus_segloss': supv_fus_segloss * self.losses_weights['supv_fus_segloss'],
        }

        return losses_dict


class Joint(nn.Module):
    def __init__(self, modeler_cfg, num_channels, num_classes, img_size):
        super(Joint, self).__init__()
        self.model = get_model(
            model_name=modeler_cfg.MODEL,
            num_channels=num_channels, num_classes=num_classes, img_size=img_size,
            **modeler_cfg.MODEL_SPECS[0]
        )
        loss_specs = modeler_cfg.LOSS_SPECS[0]
        if len(modeler_cfg.LOSS_SPECS) > 1:
            self.comploss_func = Supv_CompLoss(loss_specs, **modeler_cfg.LOSS_SPECS[1])
        else:
            self.comploss_func = Supv_CompLoss(loss_specs)

    def forward(self, image, **kwargs):
        logits_m1, logits_m2, logits_fus = self.model(image)

        losses_dict = self.comploss_func(logits_m1, logits_m2, logits_fus, kwargs["target"])

        if self.training:
            return losses_dict
        else:
            return logits_fus, losses_dict

    def get_learnable_parameters(self):
        return [v for k, v in self.model.named_parameters()]
