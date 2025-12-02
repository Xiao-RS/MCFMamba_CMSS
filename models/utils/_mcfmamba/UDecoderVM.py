import torch
import torch.nn as nn
from einops import rearrange
from models._vmamba.vmamba import VSSBlock


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H, W, C
        """

        x = self.expand(x)  # B, H, W, 2C
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale * dim_scale * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        x = self.expand(x)  # B, H, W, 16C
        B, H, W, C = x.shape
        x = rearrange(
            x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2)
        )
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = avg_out + max_out
        return x * self.sigmoid(attn)


class DecoderLayer(nn.Module):
    def __init__(self,
                 upsample,
                 dim,
                 drop_path,
                 ssm_d_state,
                 decoder_layer="DecoderVMamba_v1"):
        super().__init__()
        self.upsample = upsample
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim, drop_path=drop_path[i], d_state=ssm_d_state, forward_type="v0",
            ) for i in range(len(drop_path))])
        self.cat_conv = nn.Linear(dim * 2, dim)

    def forward(self, x, skip):
        x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        x = self.upsample(x)
        for block in self.blocks:
            x = block(x)
        x = torch.cat([x, skip.permute(0, 2, 3, 1).contiguous()], dim=-1)  # B, C, H, W
        x = self.cat_conv(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class UDecoderVM(nn.Module):
    def __init__(self,
                 num_classes=1,
                 in_channels=[96, 192, 384, 768].reverse(),
                 depths=[2, 2, 2, 2],
                 deep_supervision=False,
                 ssm_d_state=16,
                 drop_path_rate=0.1,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)  # actually only three depths are used. The last feature is simply upexpanded
        self.deep_supervision = deep_supervision

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers_up = nn.ModuleList()
        for i_layer in range(1, self.num_layers):
            self.layers_up.append(
                DecoderLayer(
                    upsample=PatchExpand(dim=in_channels[i_layer - 1]),
                    dim=in_channels[i_layer],
                    drop_path=dpr[sum(depths[:i_layer - 1]):sum(depths[:i_layer])],
                    ssm_d_state=ssm_d_state,
                )
            )
        if self.deep_supervision:
            self.up_ds_layer = FinalPatchExpand_X4(dim=in_channels[-1])
            self.out_ds_layer = nn.Conv2d(
                in_channels=in_channels[-1], out_channels=self.num_classes, kernel_size=1, bias=False
            )

        self.upsample = FinalPatchExpand_X4(dim=in_channels[-1])
        self.output = nn.Conv2d(in_channels=in_channels[-1], out_channels=self.num_classes, kernel_size=1, bias=False)

    def forward(self, features):
        x = features[-1]  # B, C, H, W
        skips = features[::-1][1:]  # [::-1]  reverse list
        for i, layer_up in enumerate(self.layers_up):
            x = layer_up(x, skips[i])
        x_last = self.upsample(x)
        x_last = self.output(x_last)
        if not self.training:
            return x_last
        if self.deep_supervision == 1:
            x_aux = self.out_ds_layer(self.up_ds_layer(features[0]))
            return x_last, x_aux
        elif self.deep_supervision == 2:
            x_aux = self.out_ds_layer(self.up_ds_layer(x))
            return x_last, x_aux
        return x_last
