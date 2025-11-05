from torch import nn
from typing import Optional
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
from models.utils._joitrinet.rewrit_deeplabv3p import DeepLabv3pDecoder
from models.utils._joitrinet.mdafm import MDAFM

class JoiTriNet(nn.Module):
    def __init__(self,
                 num_channels: dict = {'m1': 4, 'm2': 2},
                 num_classes: int = 1,
                 img_size: tuple = (256, 256),
                 ffm_name: str = "MDAFM",

                 ffm_type: str = "DecoderFusion",
                 encoder_name: str = "resnet101",
                 encoder_depth: int = 5,
                 encoder_weights: Optional[str] = "imagenet",
                 encoder_output_stride: int = 16,
                 decoder_channels: int = 256,
                 decoder_atrous_rates: tuple = (12, 24, 36),
                 activation: Optional[str] = None,
                 upsampling: int = 4,
                 ) -> None:
        super().__init__()

        self.encoder_m1 = get_encoder(
            encoder_name,
            in_channels=num_channels['m1'],
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )
        self.decoder_m1 = DeepLabv3pDecoder(
            self.encoder_m1.out_channels[-1], self.encoder_m1.out_channels[-4],
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )
        self.head_m1 = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=num_classes,
            activation=activation,
            kernel_size=3,
            upsampling=upsampling,
        )

        self.encoder_m2 = get_encoder(
            encoder_name,
            in_channels=num_channels['m2'],
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )
        self.decoder_m2 = DeepLabv3pDecoder(
            self.encoder_m2.out_channels[-1], self.encoder_m2.out_channels[-4],
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )
        self.head_m2 = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=num_classes,
            activation=activation,
            kernel_size=3,
            upsampling=upsampling,
        )

        if ffm_type == "DecoderFusion":
            self.ffm_type = "DecoderFusion"
            self.ffm = MDAFM(decoder_channels)
            self.head_fus = SegmentationHead(
                in_channels=decoder_channels,
                out_channels=num_classes,
                activation=activation,
                kernel_size=3,
                upsampling=upsampling,
            )
        elif ffm_type == "EncoderFusion":
            self.ffm_type = "EncoderFusion"
            assert self.encoder_m1.out_channels[1:] == self.encoder_m2.out_channels[1:]
            low_channels, high_channels = self.encoder_m1.out_channels[-1], self.encoder_m1.out_channels[-4]
            self.ffm_low = MDAFM(low_channels)
            self.ffm_high = MDAFM(high_channels)
            self.decoder_fus = DeepLabv3pDecoder(
                low_channels, high_channels,
                out_channels=decoder_channels,
                atrous_rates=decoder_atrous_rates,
                output_stride=encoder_output_stride,
            )

            self.head_fus = SegmentationHead(
                in_channels=decoder_channels,
                out_channels=num_classes,
                activation=activation,
                kernel_size=3,
                upsampling=upsampling,
            )

        else:
            raise ValueError("type must be EncoderFusion or DecoderFusion")

    def forward(self, image):
        features_m1 = self.encoder_m1(image['m1'])
        finalfeat_m1 = self.decoder_m1(features_m1[-1], features_m1[-4])

        features_m2 = self.encoder_m2(image['m2'])
        finalfeat_m2 = self.decoder_m2(features_m2[-1], features_m2[-4])

        if self.ffm_type == "DecoderFusion":
            finalfeat_fus = self.ffm(finalfeat_m1, finalfeat_m2)
        if self.ffm_type == "EncoderFusion":
            lowfeats_fus = self.ffm_low(features_m1[-1], features_m2[-1])
            highfeat_fus = self.ffm_high(features_m1[-4], features_m2[-4])
            finalfeat_fus = self.decoder_fus(lowfeats_fus, highfeat_fus)
        logits_fused = self.head_fus(finalfeat_fus)

        if not self.training:
            return logits_fused

        logits_m1 = self.head_m1(finalfeat_m1)
        logits_m2 = self.head_m2(finalfeat_m2)

        return logits_m1, logits_m2, logits_fused

