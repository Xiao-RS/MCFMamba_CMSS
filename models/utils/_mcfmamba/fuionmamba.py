import math
import torch
from torch import nn
from functools import partial
from models._vmamba.vmamba import mamba_init, selective_scan_fn


class MC_SS2D(nn.Module):
    # multimodal cross 2D-selective-scan
    def __init__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            k_group=8,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            **kwargs,
    ):
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # in proj =============================
        self.in_norm_m1 = nn.LayerNorm(d_model)
        self.in_norm_m2 = nn.LayerNorm(d_model)
        self.in_proj_m1 = nn.Linear(d_model, d_inner * 2, bias=False)
        self.in_proj_m2 = nn.Linear(d_model, d_inner * 2, bias=False)
        self.act = nn.SiLU()
        self.conv2d_m1 = nn.Conv2d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,
            bias=True, kernel_size=3, padding=1,
        )
        self.conv2d_m2 = nn.Conv2d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,
            bias=True, kernel_size=3, padding=1,
        )
        # x proj =============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj, A, D =============================
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=k_group,
        )

        # cat proj =============================
        self.cat_conv = nn.Sequential(
            nn.Conv2d(d_inner * 2, d_inner, groups=d_inner, kernel_size=3, stride=1, padding=1),
            self.act,
            nn.Conv2d(d_inner, d_inner, groups=d_inner, kernel_size=3, stride=1, padding=1),
            self.act,
            nn.Conv2d(d_inner, d_model, kernel_size=1),
        )

        # out proj =============================
        self.out_norm = nn.LayerNorm(d_inner)
        # self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, feat_m1, feat_m2):
        feat_m1_norm = self.in_norm_m1(feat_m1.permute(0, 2, 3, 1).contiguous())
        feat_m2_norm = self.in_norm_m2(feat_m2.permute(0, 2, 3, 1).contiguous())
        # ========== feat_m1 linear ==========
        x_m1 = self.in_proj_m1(feat_m1_norm)
        x_m1, z_m1 = x_m1.chunk(2, dim=-1)
        # ========== z_m1 attn ==========
        z_m1 = self.act(z_m1.permute(0, 3, 1, 2).contiguous())
        # ========== x_m1 Conv ==========
        x_m1 = x_m1.permute(0, 3, 1, 2).contiguous()
        x_m1 = self.act(self.conv2d_m1(x_m1))  # (b, d, h, w)
        # ========== feat_m2 linear ==========
        x_m2 = self.in_proj_m2(feat_m2_norm)
        x_m2, z_m2 = x_m2.chunk(2, dim=-1)
        # ========== z_m2 attn ==========
        z_m2 = self.act(z_m2.permute(0, 3, 1, 2).contiguous())
        # ========== x_m2 Conv ==========
        x_m2 = x_m2.permute(0, 3, 1, 2).contiguous()
        x_m2 = self.act(self.conv2d_m2(x_m2))  # (b, d, h, w)

        # ========== multimodal cross selective_scan ==========
        selective_scan = partial(selective_scan_fn, backend="mamba")

        B, D, H, W = x_m1.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W
        # ********** scan expand **********
        x_hwwh = torch.stack([
            x_m1.view(B, -1, L),
            x_m2.view(B, -1, L),
            torch.transpose(x_m1, dim0=2, dim1=3).contiguous().view(B, -1, L),
            torch.transpose(x_m2, dim0=2, dim1=3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 4, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l) k=8
        # ********** scan expand **********

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.contiguous()  # (b, k, d_state, l)
        Cs = Cs.contiguous()  # (b, k, d_state, l)

        As = -self.A_logs.float().exp()  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # ********** scan merge **********
        inv_y = torch.flip(out_y[:, 4:], dims=[-1]).view(B, 4, -1, L)
        wh_y = torch.transpose(out_y[:, 2:4].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, 2, -1, L)
        invwh_y = torch.transpose(inv_y[:, 2:4].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, 2, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y[:, 0] + invwh_y[:, 0] + \
            out_y[:, 1] + inv_y[:, 1] + wh_y[:, 1] + invwh_y[:, 1]
        # ********** scan merge **********

        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        # ========== concat ==========
        y_fus = torch.concat([
            y.permute(0, 3, 1, 2).contiguous() * z_m1,
            y.permute(0, 3, 1, 2).contiguous() * z_m2
        ], dim=1)
        y_fus = self.cat_conv(y_fus)

        # ========== out ==========
        # y_fus = self.out_proj(y_fus)
        # return y_fus
        return y_fus + feat_m1 + feat_m2


class FusionMCMF(nn.Module):
    def __init__(self, in_channels=[96, 192, 384, 768], **kwargs):
        super().__init__()
        self.fusion_layers = nn.ModuleList()
        for dim in in_channels:
            self.fusion_layers.append(MC_SS2D(dim))

    def forward(self, features):
        if isinstance(features, dict):
            feats_fus = []
            for i in range(len(features["feat_m1"])):
                feat_m1, feat_m2 = features["feat_m1"][i], features["feat_m2"][i]
                feat_fus = self.fusion_layers[i](feat_m1, feat_m2)
                # feat_fus += feat_m1 + feat_m2
                feats_fus.append(feat_fus)
        else:
            raise NotImplementedError(features)

        return feats_fus
