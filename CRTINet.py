import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchvision import models
import numpy as np

from VisionTransformer import VisionTransformer
from train_util import *
from SP_Net import SuperPixelNet
from SP_Branch import SPE_Branch, SP_Embedding
from Decoder import MDN

from einops.layers.torch import Rearrange
from functools import partial
from collections import OrderedDict
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
class Model(nn.Module):
    def __init__(self, imgsize=384, embed_dim=384, num_heads=6, patch_size=16, level_backbone=11, level_decoder=6,
                 grid_size=24):
        super(Model, self).__init__()
        self.encoder = VisionTransformer()  # vit.vit_small_patch16_384(pretrained=False)
        self.imgsize = imgsize
        self.patch_size = patch_size
        self.depth = 12

        self.SP_Net = SuperPixelNet(feature_root=32, grid_size=grid_size)
        self.SP_Embedding = SP_Embedding(in_dims=3, embed_dims=embed_dim, patch_size=patch_size, grid_size=grid_size)
        self.SP_Patch_Modelling = SPE_Branch(embed_dims=embed_dim, depth=level_backbone, grid_size=grid_size,
                                      num_heads=num_heads, proj_drop=0., drop_path_rate=0.2)
        self.feature_decoding = MDN(embed_dim=embed_dim, grid_size=grid_size, patch_size=8, depth=level_decoder)

        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss(reduction='mean')

    def forward(self, imgs_nor, labels, img_lab, xy_feat):
        B, C, H, W = imgs_nor.size()

        x_feas = self.encoder(imgs_nor)
        patch_feas = []  # 3, 6, 9, 12
        for x0 in x_feas:
            patch_feas.append(x0[:, 1:, :])

        logit_sp_softmax, img_fea192, img_fea96 = self.SP_Net(imgs_nor)

        prob = logit_sp_softmax.clone()
        assig_max, _ = torch.max(prob, dim=1, keepdim=True)
        prob = torch.where(prob == assig_max, torch.ones(prob.shape).cuda(), torch.zeros(prob.shape).cuda())

        sp_embedding = self.SP_Embedding(imgs_nor, prob)
        sp_feas = self.SP_Patch_Modelling(sp_embedding, patch_feas[1:12])
        seg_pred, assist_patch, assist_sp, assist_patch1, assist_sp1 = self.feature_decoding(img_fea192, img_fea96,patch_feas[2::2],sp_feas[1::2], prob)
        return prob, seg_pred, assist_patch, assist_sp, assist_patch1, assist_sp1
