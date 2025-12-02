import torch
import torch.nn as nn
from models._vmamba.vmamba import VSSM
from .fuionmamba import FusionMCMF


class Backbone_VSSM(VSSM):
    # Ref : https://github.com/MzeroMiko/VMamba
    # VMamba-main\classification\models\vmamba.py
    def __init__(self, in_chans,
                 out_indices=(0, 1, 2, 3), pretrained=None, norm_layer=nn.LayerNorm,  # norm_layer="ln",
                 **kwargs):
        super().__init__(
            in_chans=in_chans,
            forward_type="v0", mlp_ratio=0.0, norm_layer="ln", downsample_version="v1",
            **kwargs
        )
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        # ========== Loading Checkpoint ==========
        if pretrained:
            _ckpt = torch.load(open(kwargs['weights'], "rb"), map_location=torch.device("cpu"))["model"]
            print(f"Successfully load ckpt {kwargs['weights']}")
            for name in _ckpt.keys():
                if (name == 'stem.conv1.weight') or (name == 'patch_embed.proj.weight'):
                    if in_chans < 3:
                        _ckpt[name] = _ckpt[name][:, :in_chans]
                    if in_chans > 3:
                        _ckpt[name] = torch.cat([
                            _ckpt[name],
                            _ckpt[name].mean(dim=1, keepdim=True).repeat(1, in_chans - 3, 1, 1)
                        ], dim=1)
            incompatibleKeys = self.load_state_dict(_ckpt, strict=False)
            print('incompatible:', incompatibleKeys)

    def forward(self, x, mask=None):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)

        # ========== Masking ==========
        if mask is not None:
            mask = mask.permute(0, 2, 3, 1).contiguous()
            B, _, _, L = x.shape
            x = x * (1. - mask)
        # ========== Masking ==========

        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x

        return outs


VSSM_kwargs = {
    'VssmTiny': dict(depths=[2, 2, 9, 2], dims=96, drop_path_rate=0.2,
                     weights="E:/Code/checkpoints/vssmtiny_dp01_ckpt_epoch_292.pth"),
    'VssmSmall': dict(depths=[2, 2, 27, 2], dims=96, drop_path_rate=0.3,
                      weights="E:/Code/checkpoints/vssmsmall_dp03_ckpt_epoch_238.pth"),
    'VssmBase': dict(depths=[2, 2, 27, 2], dims=128, drop_path_rate=0.6,
                     weights="E:/Code/checkpoints/vssmbase_dp06_ckpt_epoch_241.pth"),
}


class DualVMamba(nn.Module):
    def __init__(self,
                 in_chans_m1,
                 in_chans_m2,
                 backbone_name,
                 fusion,
                 pretrained=False,
                 **kwargs):
        super().__init__()

        self.branch_m1 = Backbone_VSSM(
            in_chans=in_chans_m1, pretrained=pretrained, **VSSM_kwargs[backbone_name]
        )
        self.branch_m2 = Backbone_VSSM(
            in_chans=in_chans_m2, pretrained=pretrained, **VSSM_kwargs[backbone_name]
        )
        self.output_channels = self.branch_m1.dims
        self.fusion_module = FusionMCMF(self.output_channels)

    def forward(self, x_m1, x_m2, mask_m1=None, mask_m2=None):
        feat_m1 = self.branch_m1(x_m1, mask_m1)
        feat_m2 = self.branch_m2(x_m2, mask_m2)

        feat_fus = self.fusion_module({'feat_m1': feat_m1, 'feat_m2': feat_m2})

        return feat_fus
