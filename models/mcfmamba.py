import torch.nn as nn


class MCFMamba(nn.Module):
    def __init__(self,
                 num_channels: dict = {'m1': 3, 'm2': 1},
                 num_classes: int = 1,
                 img_size: tuple = (256, 256),

                 backbone_name: str = "VssmTiny",
                 fusion: str = "FusionMCMF",
                 decoder_name: str = "UDecoderVM",

                 pretrained: bool = True,
                 **kwargs
                 ):
        super().__init__()

        self.num_cls = num_classes
        self.compute_complexity = False
        self.num_channels_m1 = num_channels['m1']

        # import backbone and decoder
        if backbone_name.find('Vssm') == 0:
            from models.utils._mcfmamba.dual_vmamba import DualVMamba as backbone
        else:
            raise NotImplementedError(backbone_name)
        self.backbone = backbone(
            in_chans_m1=num_channels['m1'],
            in_chans_m2=num_channels['m2'],
            img_size=img_size,
            backbone_name=backbone_name,
            fusion=fusion,
            pretrained=pretrained,
        )

        if decoder_name == 'UDecoderVM':
            from models.utils._mcfmamba.UDecoderVM import UDecoderVM
            self.sem_seg_head = UDecoderVM(
                num_classes=num_classes,
                in_channels=self.backbone.output_channels[::-1],
            )
        else:
            raise NotImplementedError(decoder_name)

    def forward(self, image, masked=False):

        if masked:
            features = self.backbone(image['m1'], image['m2'], image['mask1'], image['mask2'])
        else:
            features = self.backbone(image['m1'], image['m2'])

        pred_out = self.sem_seg_head(features)
        return pred_out

