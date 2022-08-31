import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchvision import models
import numpy as np
from einops.layers.torch import Rearrange
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from train_util import poolfeat

class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0:2]).unsqueeze(1).reshape(B, 2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 2, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls


class LayerScale_Block_CA(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Class_Attention,
                 Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_cls):
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.drop_path(self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.mlp(self.norm2(x_cls)))
        return x_cls


class SP_Embedding(nn.Module):
    def __init__(self, in_dims, embed_dims=384, patch_size=16, grid_size=24):
        super(SP_Embedding, self).__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.embed_dims = embed_dims

        self.PE = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, groups=in_dims)
        self.arrange = Rearrange('b c h w -> b (h w) c')

        self.proj_embedding = nn.Sequential(
            nn.Conv2d(in_dims, embed_dims // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(embed_dims // 2),
            nn.GELU(),
            nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, prob):
        spc = poolfeat(x, prob, self.patch_size, self.patch_size)
        spc = self.proj_embedding(spc)
        pe = self.PE(spc)
        spc = spc + pe
        spc = self.arrange(spc)
        return spc

class Cross_LocalAttn(nn.Module):
    def __init__(self, embed_dims=384, drop_path=0.1, grid_size=24, num_heads=6, proj_drop=0.):
        super(Cross_LocalAttn, self).__init__()
        self.embed_dims = embed_dims
        self.grid_size = grid_size
        self.num_heads = num_heads
        self.scale = (embed_dims / num_heads * 1.0) ** -0.5
        self.h_shift = 1
        self.w_shift = 1

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.sp_qkv = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, 4 * embed_dims, bias=True)
        )
        self.sp_q = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims, bias=True)
        )
        self.proj_sp_sattn = nn.Linear(embed_dims, embed_dims, bias=True)

        self.patch_kv = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, 2 * embed_dims, bias=True)
        )
        self.proj_cross_attn = nn.Linear(embed_dims, embed_dims, bias=True)
        self.proj_cross_attn1 = nn.Linear(embed_dims, embed_dims, bias=True)

        self.attn_drop = nn.Dropout(0.)

        self.proj_mf = nn.Linear(2 * embed_dims, embed_dims, bias=True)

        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, 4 * embed_dims, bias=True),
            nn.GELU(),
            nn.Linear(4 * embed_dims, embed_dims, bias=True)
        )

        self.skip_patch = nn.Sequential(
            nn.LayerNorm(embed_dims),
            # Rearrange('b (h w) c -> b c h w', h=grid_size, w=grid_size),
            # nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, groups=embed_dims),
            # Rearrange('b c h w ->  b (h w) c', h=grid_size, w=grid_size),
            nn.Linear(embed_dims, embed_dims, bias=True)
        )

        # self.fea_align = Fea_Align(embed_dims=embed_dims, grid_size=grid_size)
        self.arrange_sp = Rearrange('b hs n (k c) -> b hs n k c', k=1)
        self.arrange_sp_back = Rearrange('b hs n k c -> b n (hs k c)')

        self.arrange_conv = Rearrange('b (h w) c -> b c h w', h=grid_size, w=grid_size)
        self.arrange_patch = Rearrange('b k (hs c l) h w -> l b hs (h w) k c', h=grid_size, w=grid_size, hs=num_heads,
                                       l=2)

    def forward(self, fea_sp, fea_patch):
        B, N, C = fea_sp.shape
        sp_qkv = self.sp_qkv(fea_sp)
        sp_qkv = sp_qkv.reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        sp_q, sp_q1, sp_k, sp_v = sp_qkv[0], sp_qkv[1], sp_qkv[2], sp_qkv[3]

        sp_sattn = (sp_q @ sp_k.transpose(-2, -1)) * self.scale
        sp_sattn = sp_sattn.softmax(dim=-1)
        sp_sattn = self.attn_drop(sp_sattn)
        sp_sattn = (sp_sattn @ sp_v).transpose(1, 2).reshape(B, N, C)
        sp_sattn = self.proj_sp_sattn(sp_sattn)

        patch_kv = self.patch_kv(fea_patch)
        patch_kv = self.arrange_conv(patch_kv)
        patch_kv = patch_kv
        kv_pad = F.pad(patch_kv, (1, 1, 1, 1), mode='replicate')

        kv1 = kv_pad[:, :, :-2 * self.h_shift, :-2 * self.w_shift].unsqueeze(1)
        kv2 = kv_pad[:, :, :-2 * self.h_shift, self.w_shift:-self.w_shift].unsqueeze(1)
        kv3 = kv_pad[:, :, :-2 * self.h_shift, 2 * self.w_shift:].unsqueeze(1)
        kv4 = kv_pad[:, :, self.h_shift:-self.w_shift, :-2 * self.w_shift].unsqueeze(1)
        kv5 = patch_kv.unsqueeze(1)
        kv6 = kv_pad[:, :, self.h_shift:-self.w_shift, 2 * self.w_shift:].unsqueeze(1)
        kv7 = kv_pad[:, :, 2 * self.h_shift:, :-2 * self.w_shift].unsqueeze(1)
        kv8 = kv_pad[:, :, 2 * self.h_shift:, self.w_shift:-self.w_shift].unsqueeze(1)
        kv9 = kv_pad[:, :, 2 * self.h_shift:, 2 * self.w_shift:].unsqueeze(1)

        patch_kv_merge = torch.cat([kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9], dim=1) \
            .view(-1, 9, 2, self.num_heads, self.embed_dims // self.num_heads, self.grid_size * self.grid_size) \
            .permute(2, 0, 5, 3, 1, 4)  # num_kv, batch, N, num_heads, 9, dim
        patch_k, patch_v = patch_kv_merge[0], patch_kv_merge[1]

        sp_q = sp_q1.view(-1, self.num_heads, self.grid_size * self.grid_size, 1,
                          self.embed_dims // self.num_heads).transpose(2, 1)
        attn = (sp_q @ patch_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        proj_cross_attn = (attn @ patch_v).transpose(1, 2).reshape(B, N, C)
        sp_patch_attn = self.proj_cross_attn(proj_cross_attn)

        x = torch.cat([sp_sattn, sp_patch_attn], dim=-1)
        x = self.drop_path(self.proj_mf(x))
        x = fea_sp + x
        x = x + self.drop_path(self.ffn(x))
        return x


class SP_GLOBAL_Attn(nn.Module):
    def __init__(self, embed_dims=384, drop_path=0.1, grid_size=24, num_heads=6, proj_drop=0.):
        super(SP_GLOBAL_Attn, self).__init__()
        self.embed_dims = embed_dims
        self.grid_size = grid_size
        self.num_heads = num_heads
        self.scale = (embed_dims / num_heads * 1.0) ** -0.5
        self.h_shift = 1
        self.w_shift = 1

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sp_qkv = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, 3 * embed_dims, bias=True)
        )
        self.proj_sp_attn = nn.Linear(embed_dims, embed_dims, bias=True)

        self.attn_drop = nn.Dropout(0.)

        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, 4 * embed_dims, bias=True),
            nn.GELU(),
            nn.Linear(4 * embed_dims, embed_dims, bias=True)
        )

        self.arrange_sp = Rearrange('b hs n (k c) -> b hs n k c', k=1)
        self.arrange_sp_back = Rearrange('b hs n k c -> b n (hs k c)')

        self.arrange_conv = Rearrange('b (h w) c -> b c h w', h=grid_size, w=grid_size)
        self.arrange_patch = Rearrange('b k (hs c l) h w -> l b hs (h w) k c', h=grid_size, w=grid_size, hs=num_heads,
                                       l=2)

    def forward(self, fea_sp):
        B, N, C = fea_sp.shape
        qkv = self.sp_qkv(fea_sp).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        attn = self.proj_sp_attn(attn)

        x = fea_sp + self.drop_path(attn)
        x = x + self.drop_path(self.ffn(x))
        return x


class SPE_Branch(nn.Module):
    def __init__(self, embed_dims=768, depth=6, grid_size=14, num_heads=6, proj_drop=0., drop_path_rate=0.1):
        super().__init__()
        self.depth = depth

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.modulelist_spt_attn = nn.ModuleList()
        for i in range(depth):
            self.modulelist_spt_attn.append(
                Cross_LocalAttn(embed_dims=embed_dims, drop_path=dpr[i], grid_size=grid_size, num_heads=num_heads,
                                proj_drop=proj_drop)
            )

        self.gattn = SP_GLOBAL_Attn(embed_dims=embed_dims, drop_path=0., grid_size=grid_size, num_heads=num_heads,
                                    proj_drop=proj_drop)
    def forward(self, fea_sp, fea_patch):
        features_sp = []

        fea_sp = self.gattn(fea_sp)
        features_sp.append(fea_sp)

        for i in range(self.depth):
            fea_sp = self.modulelist_spt_attn[i](fea_sp, fea_patch[i])
            features_sp.append(fea_sp)
        return features_sp


def compute_feature_pos_loss(prob_in, labxy_feat, pos_weight=0.003, kernel_size=16):
    # this wrt the slic paper who used sqrt of (mse)
    # rgbxy1_feat: B*50+2*H*W
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure
    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    difference_fea = reconstr_feat[:, :-2, :, :] - labxy_feat[:, :-2, :, :]
    difference_pos = reconstr_feat[:, -2:, :, :] - labxy_feat[:, -2:, :, :]

    loss_sem = torch.mean(torch.mean(difference_fea * difference_fea, dim=(1, 2, 3)))
    loss_pos = torch.norm(difference_pos, p=2, dim=1).sum() / b * m / S

    # empirically we find timing 0.005 tend to better performance
    loss_sum = 0.005 * (loss_sem + loss_pos)
    # loss_sem_sum =  0.005*loss_sem
    # loss_pos_sum = 0.005*loss_pos
    return loss_sum


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
