import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchvision import models
import numpy as np
from einops.layers.torch import Rearrange
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_


def upfeat(input, prob, up_h=2, up_w=2):
    # input b*n*H*W  downsampled
    # prob b*9*h*w
    b, c, h, w = input.shape

    h_shift = 1
    w_shift = 1

    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(input, p2d, mode='constant', value=0)

    gt_frm_top_left = F.interpolate(feat_pd[:, :, :-2 * h_shift, :-2 * w_shift], size=(h * up_h, w * up_w),
                                    mode='nearest')
    feat_sum = gt_frm_top_left * prob.narrow(1, 0, 1)

    top = F.interpolate(feat_pd[:, :, :-2 * h_shift, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top * prob.narrow(1, 1, 1)

    top_right = F.interpolate(feat_pd[:, :, :-2 * h_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top_right * prob.narrow(1, 2, 1)

    left = F.interpolate(feat_pd[:, :, h_shift:-w_shift, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += left * prob.narrow(1, 3, 1)

    center = F.interpolate(input, (h * up_h, w * up_w), mode='nearest')
    feat_sum += center * prob.narrow(1, 4, 1)

    right = F.interpolate(feat_pd[:, :, h_shift:-w_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += right * prob.narrow(1, 5, 1)

    bottom_left = F.interpolate(feat_pd[:, :, 2 * h_shift:, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_left * prob.narrow(1, 6, 1)

    bottom = F.interpolate(feat_pd[:, :, 2 * h_shift:, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom * prob.narrow(1, 7, 1)

    bottom_right = F.interpolate(feat_pd[:, :, 2 * h_shift:, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_right * prob.narrow(1, 8, 1)
    return feat_sum


class SP_upSampling(nn.Module):
    def __init__(self, in_dims=32, patch_size=8, grid_size=24, is_convReshape=False):
        super(SP_upSampling, self).__init__()
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.in_dims = in_dims
        self.w_shift = 1
        self.h_shift = 1
        self.is_convReshape = is_convReshape

        self.local_arrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2) c', h=grid_size, w=grid_size,
                                       p1=patch_size, p2=patch_size)
        self.local_rearrange = Rearrange('b (h w) (p1 p2) c -> b c (h p1) (w p2)', h=grid_size, w=grid_size,
                                         p1=patch_size, p2=patch_size)

    def forward(self, fea, prob):
        if self.is_convReshape == False:
            shapes = fea.shape
            fea = fea.transpose(-1, -2).reshape(-1, shapes[2], self.grid_size, self.grid_size)
        attn = self.local_arrange(prob)
        kv = fea
        kv_pad = F.pad(kv, (1, 1, 1, 1), mode='replicate')
        kv1 = kv_pad[:, :, :-2 * self.h_shift, :-2 * self.w_shift].unsqueeze(1)
        kv2 = kv_pad[:, :, :-2 * self.h_shift, self.w_shift:-self.w_shift].unsqueeze(1)
        kv3 = kv_pad[:, :, :-2 * self.h_shift, 2 * self.w_shift:].unsqueeze(1)
        kv4 = kv_pad[:, :, self.h_shift:-self.w_shift, :-2 * self.w_shift].unsqueeze(1)
        kv5 = kv.unsqueeze(1)
        kv6 = kv_pad[:, :, self.h_shift:-self.w_shift, 2 * self.w_shift:].unsqueeze(1)
        kv7 = kv_pad[:, :, 2 * self.h_shift:, :-2 * self.w_shift].unsqueeze(1)
        kv8 = kv_pad[:, :, 2 * self.h_shift:, self.w_shift:-self.w_shift].unsqueeze(1)
        kv9 = kv_pad[:, :, 2 * self.h_shift:, 2 * self.w_shift:].unsqueeze(1)
        kv_merge = torch.cat([kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9], dim=1).view(-1, 9, self.in_dims,
                                                                                        self.grid_size * self.grid_size).permute(
            0, 3, 1, 2)
        attn = attn @ kv_merge
        attn = self.local_rearrange(attn)
        return attn


def LTA(fea, h_shift=1, w_shift=1, num_heads=3, embed_dims=384, grid_size=24):
    fea_pad = F.pad(fea, (1, 1, 1, 1), mode='replicate')
    v1 = fea_pad[:, :, :-2 * h_shift, :-2 * w_shift].unsqueeze(1)
    v2 = fea_pad[:, :, :-2 * h_shift, w_shift:-w_shift].unsqueeze(1)
    v3 = fea_pad[:, :, :-2 * h_shift, 2 * w_shift:].unsqueeze(1)
    v4 = fea_pad[:, :, h_shift:-w_shift, :-2 * w_shift].unsqueeze(1)
    v5 = fea.unsqueeze(1)
    v6 = fea_pad[:, :, h_shift:-w_shift, 2 * w_shift:].unsqueeze(1)
    v7 = fea_pad[:, :, 2 * h_shift:, :-2 * w_shift].unsqueeze(1)
    v8 = fea_pad[:, :, 2 * h_shift:, w_shift:-w_shift].unsqueeze(1)
    v9 = fea_pad[:, :, 2 * h_shift:, 2 * w_shift:].unsqueeze(1)

    v_merge = torch.cat([v1, v2, v3, v4, v5, v6, v7, v8, v9], dim=1) \
        .view(-1, 9, num_heads, embed_dims // num_heads, grid_size * grid_size) \
        .permute(0, 4, 2, 1, 3)  # num_kv, batch, N, num_heads, 9, dim
    return v_merge


class Attn_Decoder(nn.Module):  # Attn_Decoder
    def __init__(self, patch_indims=96, sp_indims=384, embed_dims=96, drop_path=0.1, grid_size=24, num_heads=2,
                 patch_size=8, proj_drop=0.):
        super(Attn_Decoder, self).__init__()
        self.embed_dims = embed_dims
        self.grid_size = grid_size
        self.num_heads = 12  # num_heads
        self.scale = (embed_dims / num_heads * 1.0) ** -0.5
        self.patch_size = patch_size
        self.h_shift = 1
        self.w_shift = 1
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.proj_temp1 = nn.Sequential(
            nn.Conv2d(patch_indims, 9 * self.num_heads, kernel_size=1, stride=1, padding=0),
            Rearrange('b (c head) (h p1) (w p2) -> b (h w) head (p1 p2) c',
                      h=grid_size, w=grid_size, p1=self.patch_size, p2=self.patch_size, head=self.num_heads)
        )
        self.proj_temp2 = nn.Sequential(
            nn.Conv2d(patch_indims, 9 * self.num_heads, kernel_size=1, stride=1, padding=0),
            Rearrange('b (c head) (h p1) (w p2) -> b (h w) head (p1 p2) c',
                      h=grid_size, w=grid_size, p1=self.patch_size, p2=self.patch_size, head=self.num_heads)
        )

        self.proj_patch = nn.Sequential(
            nn.LayerNorm(sp_indims),
            nn.Linear(sp_indims, embed_dims, bias=True),
            Rearrange('b (h w) c -> b c h w', h=grid_size, w=grid_size)
        )
        self.proj_sp = nn.Sequential(
            nn.LayerNorm(sp_indims),
            nn.Linear(sp_indims, embed_dims, bias=True),
            Rearrange('b (h w) c -> b c h w', h=grid_size, w=grid_size)
        )

        self.proj_fpatch = nn.Sequential(
            nn.BatchNorm2d(patch_indims),
            nn.Conv2d(patch_indims, patch_indims, kernel_size=1, stride=1, padding=0)
        )

        self.proj_fsp = nn.Sequential(
            nn.BatchNorm2d(patch_indims),
            nn.Conv2d(patch_indims, patch_indims, kernel_size=1, stride=1, padding=0)
        )

        self.mf_patch = nn.Sequential(
            nn.Conv2d(2 * patch_indims, patch_indims, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(patch_indims),
            nn.ReLU(inplace=True),
            nn.Conv2d(patch_indims, patch_indims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(patch_indims),
        )

        self.mf_sp = nn.Sequential(
            nn.Conv2d(2 * patch_indims, patch_indims, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(patch_indims),
            nn.ReLU(inplace=True),
            nn.Conv2d(patch_indims, patch_indims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(patch_indims)
        )

        self.arrange_sp = Rearrange('b (h w) head (p1 p2) c -> b (c head) (h p1) (w p2)', h=grid_size, w=grid_size,
                                    p1=self.patch_size, p2=self.patch_size, head=self.num_heads)
        self.attn_drop = nn.Dropout(0.)
        self.ffn = nn.Sequential(
            nn.Conv2d(patch_indims, patch_indims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(patch_indims),
            nn.ReLU(inplace=True),
            nn.Conv2d(patch_indims, patch_indims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(patch_indims),
            nn.ReLU(inplace=True)
        )

    def forward(self, temp_fea, temp_sp, temp_patch, previous_patch=0., previous_sp=0.):
        proj_temp1 = self.proj_temp1(temp_fea)

        # ********** begin EII Module
        proj_patch = self.proj_patch(temp_patch)
        patch_v = LTA(proj_patch, h_shift=self.h_shift, w_shift=self.w_shift, num_heads=self.num_heads,
                      embed_dims=self.embed_dims, grid_size=self.grid_size)
        fea_patch = (proj_temp1 @ patch_v)  # batch, N, num_heads, patchsize, dim
        fea_patch = self.arrange_sp(fea_patch)
        fea_patch = self.proj_fpatch(fea_patch) + previous_patch
        fea_patch = self.mf_patch(torch.cat([fea_patch, temp_fea], dim=1))
        # ********** end EII Module

        # ********** begin TII Module
        proj_temp2 = self.proj_temp2(temp_fea)
        proj_sp = self.proj_sp(temp_sp)
        sp_v = LTA(proj_sp, h_shift=self.h_shift, w_shift=self.w_shift, num_heads=self.num_heads,
                   embed_dims=self.embed_dims, grid_size=self.grid_size)
        fea_sp = (proj_temp2 @ sp_v)  # batch, N, num_heads, patchsize, dim
        fea_sp = self.arrange_sp(fea_sp)
        fea_sp = self.proj_fsp(fea_sp) + previous_sp
        fea_sp = self.mf_sp(torch.cat([fea_sp, temp_fea], dim=1))
        # ********** end TII Module

        # ********** begin fusion Module
        x = self.ffn(fea_patch + fea_sp)
        # ********** end fusion Module

        return x, fea_patch, fea_sp


class Fea_Decoder(nn.Module):
    def __init__(self, embed_dim=768, depth=6):
        super(Fea_Decoder, self).__init__()
        self.depth = depth
        self.modulelist_mf_fea = nn.ModuleList()
        for i in range(self.depth):
            self.modulelist_mf_fea.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True)
                )
            )

        self.modulelist_proj_patch = nn.ModuleList()
        for i in range(self.depth):
            self.modulelist_proj_patch.append(
                nn.Sequential(
                    nn.Conv2d(2 * embed_dim, embed_dim, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(embed_dim)
                )
            )

        self.modulelist_proj_sp = nn.ModuleList()
        for i in range(self.depth):
            self.modulelist_proj_sp.append(
                nn.Sequential(
                    nn.Conv2d(2 * embed_dim, embed_dim, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(embed_dim)
                )
            )

    def forward(self, temp_fea, fea_patch, fea_sp):
        mf_fea = []
        patch_logits = []
        sp_logits = []
        for i in range(self.depth):
            temp1 = self.modulelist_proj_patch[i](torch.cat([temp_fea, fea_patch[i]], dim=1))
            temp2 = self.modulelist_proj_sp[i](torch.cat([temp_fea, fea_sp[i]], dim=1))
            temp_fea = self.modulelist_mf_fea[i](temp1 + temp2)
            mf_fea.append(temp_fea)
            patch_logits.append(temp1)
            sp_logits.append(temp2)
        return mf_fea, patch_logits, sp_logits


class MDN(nn.Module):
    def __init__(self, embed_dim=768, grid_size=24, patch_size=8, depth=6):
        super(MDN, self).__init__()
        self.grid_size = grid_size
        self.depth = depth
        embed_dim = 96

        # *************begin superpixel direct upsampling (UM)
        self.modulelist_proj_sp = nn.ModuleList()
        for i in range(self.depth):
            self.modulelist_proj_sp.append(
                nn.Sequential(
                    nn.LayerNorm(384),
                    nn.Linear(384, embed_dim, bias=True),
                    Rearrange('b (h w) c -> b c h w', h=grid_size, w=grid_size)
                )
            )

        self.modulelist_upsp = nn.ModuleList()
        for i in range(self.depth):
            self.modulelist_upsp.append(
                SP_upSampling(in_dims=embed_dim, patch_size=4, grid_size=grid_size, is_convReshape=True)
            )

        self.modulelist_proj_sp1 = nn.ModuleList()
        for i in range(self.depth):
            self.modulelist_proj_sp1.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(embed_dim)
                )
            )
        # *************end superpixel direct upsampling

        # *************begin patch direct upsampling
        self.modulelist_up_patch = nn.ModuleList()
        for i in range(self.depth):
            self.modulelist_up_patch.append(
                nn.Sequential(
                    nn.LayerNorm(384),
                    nn.Linear(384, 4 * embed_dim, bias=True),
                    Rearrange('b (h w) c -> b c h w', h=grid_size, w=grid_size),
                    nn.PixelShuffle(upscale_factor=2),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(embed_dim)
                )
            )
        # *************end patch direct upsampling

        # *************begin patch/sp self-adaptive upsampling
        dpr = [x.item() for x in torch.linspace(0, 0.2, depth)]
        self.attn_decoder = nn.ModuleList()
        for i in range(self.depth):
            self.attn_decoder.append(
                Attn_Decoder(patch_indims=embed_dim, sp_indims=384, embed_dims=embed_dim, drop_path=dpr[i],
                             grid_size=24, num_heads=3, patch_size=4, proj_drop=0.0)
            )
        # begin *************end patch/sp self-adaptive upsampling

        # dpr = [x.item() for x in torch.linspace(0, 0.2, depth)]
        # self.attn_decoder1 = nn.ModuleList()
        # for i in range(self.depth):
        #     self.attn_decoder1.append(
        #         Attn_Decoder(patch_indims=embed_dim, sp_indims=384, embed_dims=embed_dim,
        #                      drop_path=dpr[i], grid_size=24, num_heads=3, patch_size=8, proj_drop=0.0)
        #     )

        self.fea_decoder = Fea_Decoder(embed_dim=embed_dim, depth=depth)

        self.modulelist_up_patch1 = nn.ModuleList()
        for i in range(self.depth):
            self.modulelist_up_patch1.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2)
                )
            )

        self.modulelist_up_sp1 = nn.ModuleList()
        for i in range(self.depth):
            self.modulelist_up_sp1.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2)
                )
            )

        self.proj_fea96 = nn.Sequential(
            nn.Conv2d(256, embed_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.proj_fea192 = nn.Sequential(
            nn.Conv2d(64, embed_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.seg_pred = nn.Sequential(
            nn.Conv2d(self.depth * embed_dim, embed_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(embed_dim, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.patch_pred = nn.Sequential(
            nn.Conv2d(embed_dim, 1, kernel_size=1, stride=1, padding=0),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Sigmoid()
        )

        self.sp_logits = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1, bias=True),
            Rearrange('b (h w) c -> b c h w', h=grid_size, w=grid_size)
        )

        self.sp_logits_up = SP_upSampling(in_dims=1, patch_size=16, grid_size=grid_size, is_convReshape=True)
        self.sigmoid = nn.Sigmoid()

        self.modulelist_pred_logits = nn.ModuleList()
        for i in range(self.depth):
            self.modulelist_pred_logits.append(
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(embed_dim)
                )
            )

        self.modulelist_sp_pred = nn.ModuleList()
        for i in range(self.depth):
            self.modulelist_sp_pred.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, 1, kernel_size=1, stride=1, padding=0),
                    nn.UpsamplingBilinear2d(scale_factor=4),
                    nn.Sigmoid()
                )
            )

        self.modulelist_patch_pred = nn.ModuleList()
        for i in range(self.depth):
            self.modulelist_patch_pred.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, 1, kernel_size=1, stride=1, padding=0),
                    nn.UpsamplingBilinear2d(scale_factor=4),
                    nn.Sigmoid()
                )
            )

        self.modulelist_sp_pred1 = nn.ModuleList()
        for i in range(self.depth):
            self.modulelist_sp_pred1.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, 1, kernel_size=1, stride=1, padding=0),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Sigmoid()
                )
            )

        self.modulelist_patch_pred1 = nn.ModuleList()
        for i in range(self.depth):
            self.modulelist_patch_pred1.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, 1, kernel_size=1, stride=1, padding=0),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Sigmoid()
                )
            )

    def forward(self, img_fea192, img_fea96, patch_feas, spc_feas, prob):
        proj_sp = []
        for i in range(self.depth):
            temp = self.modulelist_proj_sp[i](spc_feas[i])
            proj_sp.append(temp)

        up_sp = []
        for i in range(self.depth):
            temp = self.modulelist_proj_sp1[i](self.modulelist_upsp[i](proj_sp[i], prob[:, :, 0::4, 0::4]))
            up_sp.append(temp)

        up_patch = []
        for i in range(self.depth):
            temp = self.modulelist_up_patch[i](patch_feas[i])
            up_patch.append(temp)

        fea_decoder = []
        pt_logits = []
        sp_logits = []
        temp_fea = self.proj_fea96(img_fea96)
        temp = temp_fea
        for i in range(1, self.depth + 1):
            temp, temp_pt, temp_sp = self.attn_decoder[i - 1](temp, spc_feas[-i], patch_feas[-i],
                                                              previous_patch=up_patch[-i], previous_sp=up_sp[-i])  #
            fea_decoder.append(temp)
            pt_logits.append(temp_pt)
            sp_logits.append(temp_sp)

        up_patch1 = []
        for i in range(self.depth):
            temp = self.modulelist_up_patch1[i](pt_logits[i])
            up_patch1.append(temp)

        up_sp1 = []
        for i in range(self.depth):
            temp = self.modulelist_up_sp1[i](sp_logits[i])
            up_sp1.append(temp)

        temp_fea = self.proj_fea192(img_fea192)
        fea_decoder1, pt_logits1, sp_logits1 = self.fea_decoder(temp_fea, up_patch1, up_sp1)

        pred_logits = []
        for i in range(self.depth):
            temp = self.modulelist_pred_logits[i](fea_decoder[i]) + fea_decoder1[i]
            pred_logits.append(temp)

        patch_pred = []
        for i in range(self.depth):
            temp = self.modulelist_patch_pred[i](pt_logits[i])
            patch_pred.append(temp)

        sp_pred = []
        for i in range(self.depth):
            temp = self.modulelist_sp_pred[i](sp_logits[i])
            sp_pred.append(temp)

        patch_pred1 = []
        for i in range(self.depth):
            temp = self.modulelist_patch_pred1[i](pt_logits1[i])
            patch_pred1.append(temp)

        sp_pred1 = []
        for i in range(self.depth):
            temp = self.modulelist_sp_pred1[i](sp_logits1[i])
            sp_pred1.append(temp)

        seg_pred = self.seg_pred(torch.cat(pred_logits, dim=1))
        return seg_pred, patch_pred, sp_pred, patch_pred1, sp_pred1
