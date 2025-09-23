# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath, trunc_normal_



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



# HMSA_coder_1
class Attention_HMSA_decoder_1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 权重初始化函数，对线性层和层归一化层进行不同的初始化
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map=False):
        # 获取输入张量的形状信息
        B, N, C = x.shape
        # 使用线性映射计算查询（q）、键（k）、和值（v）
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 计算注意力分数，并进行缩放
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        # 对注意力分数进行 softmax 操作，并取平均
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N ** .5)
        # 计算注意力分数与位置之间的关系
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        distances = indd ** .5
        distances = distances.to('cuda')

        # 计算位置与注意力分数的加权平均距离
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= N

        if return_map:
            return dist, attn_map
        else:
            return dist

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



# rgb self attention(MHSA)
class Attention_1_convertor_rgb_MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N ** .5)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        distances = indd ** .5
        distances = distances.to('cuda')

        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= N

        if return_map:
            return dist, attn_map
        else:
            return dist



    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# depth self attention(MHSA)
class Attention_2_convertor_depth_MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N ** .5)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        distances = indd ** .5
        distances = distances.to('cuda')

        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= N

        if return_map:
            return dist, attn_map
        else:
            return dist

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class MutualAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.rgb_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_proj = nn.Linear(dim, dim)

        self.depth_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, rgb_fea, depth_fea):
        B, N, C = rgb_fea.shape

        rgb_q = self.rgb_q(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        rgb_k = self.rgb_k(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        rgb_v = self.rgb_v(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q [B, nhead, N, C//nhead]

        depth_q = self.depth_q(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_k = self.depth_k(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_v = self.depth_v(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # rgb branch
        rgb_attn = (rgb_q @ depth_k.transpose(-2, -1)) * self.scale
        rgb_attn = rgb_attn.softmax(dim=-1)
        rgb_attn = self.attn_drop(rgb_attn)

        rgb_fea = (rgb_attn @ depth_v).transpose(1, 2).reshape(B, N, C)
        rgb_fea = self.rgb_proj(rgb_fea)
        rgb_fea = self.proj_drop(rgb_fea)

        # depth branch
        depth_attn = (depth_q @ rgb_k.transpose(-2, -1)) * self.scale
        depth_attn = depth_attn.softmax(dim=-1)
        depth_attn = self.attn_drop(depth_attn)

        depth_fea = (depth_attn @ rgb_v).transpose(1, 2).reshape(B, N, C)
        depth_fea = self.depth_proj(depth_fea)
        depth_fea = self.proj_drop(depth_fea)

        return rgb_fea, depth_fea


class Mutual_Cross_Agent_Attention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, sr_ratio_rgb=1, sr_ratio_depth=1, agent_num=49, rgb_agent_num=49, depth_agent_num=49):#, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # rgb qkv
        self.rgb_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.rgb_attn_drop = nn.Dropout(attn_drop)
        self.rgb_proj = nn.Linear(dim, dim)
        self.rgb_proj_drop = nn.Dropout(proj_drop)

        # depth qkv
        self.depth_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.depth_attn_drop = nn.Dropout(attn_drop)
        self.depth_proj = nn.Linear(dim, dim)
        self.depth_proj_drop = nn.Dropout(proj_drop)

        # 1.rgb
        self.sr_ratio_rgb = sr_ratio_rgb
        if sr_ratio_rgb > 1:
            self.sr_rgb = nn.Conv2d(dim, dim, kernel_size=sr_ratio_rgb, stride=sr_ratio_rgb)
            self.norm_rgb = nn.LayerNorm(dim)
        # 2.depth
        self.sr_ratio_depth = sr_ratio_depth
        if sr_ratio_depth > 1:
            self.sr_depth = nn.Conv2d(dim, dim, kernel_size=sr_ratio_depth, stride=sr_ratio_depth)
            self.norm_depth = nn.LayerNorm(dim)

        # 1.rgb
        self.rgb_agent_num = rgb_agent_num
        self.rgb_dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.rgb_an_bias = nn.Parameter(torch.zeros(num_heads, rgb_agent_num, 7, 7))
        self.rgb_na_bias = nn.Parameter(torch.zeros(num_heads, rgb_agent_num, 7, 7))
        self.rgb_ah_bias = nn.Parameter(torch.zeros(1, num_heads, rgb_agent_num, window_size[0] // sr_ratio_rgb, 1))
        self.rgb_aw_bias = nn.Parameter(torch.zeros(1, num_heads, rgb_agent_num, 1, window_size[1] // sr_ratio_rgb))
        self.rgb_ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, rgb_agent_num))
        self.rgb_wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], rgb_agent_num))
        trunc_normal_(self.rgb_an_bias, std=.02)
        trunc_normal_(self.rgb_na_bias, std=.02)
        trunc_normal_(self.rgb_ah_bias, std=.02)
        trunc_normal_(self.rgb_aw_bias, std=.02)
        trunc_normal_(self.rgb_ha_bias, std=.02)
        trunc_normal_(self.rgb_wa_bias, std=.02)
        rgb_pool_size = int(rgb_agent_num ** 0.5)
        self.rgb_pool = nn.AdaptiveAvgPool2d(output_size=(rgb_pool_size, rgb_pool_size))
        # 2.depth
        self.depth_agent_num = depth_agent_num
        self.depth_dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.depth_an_bias = nn.Parameter(torch.zeros(num_heads, depth_agent_num, 7, 7))
        self.depth_na_bias = nn.Parameter(torch.zeros(num_heads, depth_agent_num, 7, 7))
        self.depth_ah_bias = nn.Parameter(torch.zeros(1, num_heads, depth_agent_num, window_size[0] // sr_ratio_depth, 1))
        self.depth_aw_bias = nn.Parameter(torch.zeros(1, num_heads, depth_agent_num, 1, window_size[1] // sr_ratio_depth))
        self.depth_ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, depth_agent_num))
        self.depth_wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], depth_agent_num))
        trunc_normal_(self.depth_an_bias, std=.02)
        trunc_normal_(self.depth_na_bias, std=.02)
        trunc_normal_(self.depth_ah_bias, std=.02)
        trunc_normal_(self.depth_aw_bias, std=.02)
        trunc_normal_(self.depth_ha_bias, std=.02)
        trunc_normal_(self.depth_wa_bias, std=.02)
        depth_pool_size = int(depth_agent_num ** 0.5)
        self.depth_pool = nn.AdaptiveAvgPool2d(output_size=(depth_pool_size, depth_pool_size))

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, rgb_fea, depth_fea, H=14, W=14):
        b, n, c = rgb_fea.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        rgb_q = self.rgb_q(rgb_fea)
        depth_q = self.depth_q(depth_fea)

        # 1.rgb
        if self.sr_ratio_rgb > 1:
            rgb_fea_ = rgb_fea.permute(0, 2, 1).reshape(b, c, H, W)
            rgb_fea_ = self.sr(rgb_fea_).reshape(b, c, -1).permute(0, 2, 1)
            rgb_fea_ = self.norm(rgb_fea_)
            rgb_kv = self.rgb_kv(rgb_fea_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            rgb_kv = self.rgb_kv(rgb_fea).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        rgb_k, rgb_v = rgb_kv[0], rgb_kv[1]
        # 2.depth
        if self.sr_ratio_depth > 1:
            depth_fea_ = depth_fea.permute(0, 2, 1).reshape(b, c, H, W)
            depth_fea_ = self.sr(depth_fea_).reshape(b, c, -1).permute(0, 2, 1)
            depth_fea_ = self.norm(depth_fea_)
            depth_kv = self.depth_kv(depth_fea_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            depth_kv = self.depth_kv(depth_fea).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        depth_k, depth_v = depth_kv[0], depth_kv[1]

        # get agent tokens
        # 1.rgb_agent_tokens
        rgb_agent_tokens = self.rgb_pool(rgb_q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        rgb_q = rgb_q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        rgb_k = rgb_k.reshape(b, n // self.sr_ratio_rgb ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        rgb_v = rgb_v.reshape(b, n // self.sr_ratio_rgb ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        rgb_agent_tokens = rgb_agent_tokens.reshape(b, self.rgb_agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        # 2.depth_agent_tokens
        depth_agent_tokens = self.depth_pool(depth_q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        depth_q = depth_q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        depth_k = depth_k.reshape(b, n // self.sr_ratio_depth ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        depth_v = depth_v.reshape(b, n // self.sr_ratio_depth ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        depth_agent_tokens = depth_agent_tokens.reshape(b, self.depth_agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # 1.rgb
        rgb_kv_size = (self.window_size[0] // self.sr_ratio_rgb, self.window_size[1] // self.sr_ratio_rgb)
        rgb_position_bias1 = nn.functional.interpolate(self.rgb_an_bias, size=rgb_kv_size, mode='bilinear')
        rgb_position_bias1 = rgb_position_bias1.reshape(1, num_heads, self.rgb_agent_num, -1).repeat(b, 1, 1, 1)
        rgb_position_bias2 = (self.rgb_ah_bias + self.rgb_aw_bias).reshape(1, num_heads, self.rgb_agent_num, -1).repeat(b, 1, 1, 1)
        rgb_position_bias = rgb_position_bias1 + rgb_position_bias2
        # 2.depth
        depth_kv_size = (self.window_size[0] // self.sr_ratio_depth, self.window_size[1] // self.sr_ratio_depth)
        depth_position_bias1 = nn.functional.interpolate(self.depth_an_bias, size=depth_kv_size, mode='bilinear')
        depth_position_bias1 = depth_position_bias1.reshape(1, num_heads, self.depth_agent_num, -1).repeat(b, 1, 1, 1)
        depth_position_bias2 = (self.depth_ah_bias + self.depth_aw_bias).reshape(1, num_heads, self.depth_agent_num, -1).repeat(b, 1, 1, 1)
        depth_position_bias = depth_position_bias1 + depth_position_bias2

        # first cross agent
        depth_agent_attn = self.softmax((depth_agent_tokens * self.scale) @ rgb_k.transpose(-2, -1) + rgb_position_bias)
        depth_agent_attn = self.depth_attn_drop(depth_agent_attn)
        depth_agent_v = depth_agent_attn @ rgb_v
        rgb_agent_attn = self.softmax((rgb_agent_tokens * self.scale) @ depth_k.transpose(-2, -1) + depth_position_bias)
        rgb_agent_attn = self.rgb_attn_drop(rgb_agent_attn)
        rgb_agent_v = rgb_agent_attn @ depth_v

        # 1.rgb
        rgb_agent_bias1 = nn.functional.interpolate(self.rgb_na_bias, size=self.window_size, mode='bilinear')
        rgb_agent_bias1 = rgb_agent_bias1.reshape(1, num_heads, self.rgb_agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        rgb_agent_bias2 = (self.rgb_ha_bias + self.rgb_wa_bias).reshape(1, num_heads, -1, self.rgb_agent_num).repeat(b, 1, 1, 1)
        rgb_agent_bias = rgb_agent_bias1 + rgb_agent_bias2
        # 2.depth
        depth_agent_bias1 = nn.functional.interpolate(self.depth_na_bias, size=self.window_size, mode='bilinear')
        depth_agent_bias1 = depth_agent_bias1.reshape(1, num_heads, self.depth_agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        depth_agent_bias2 = (self.depth_ha_bias + self.depth_wa_bias).reshape(1, num_heads, -1, self.depth_agent_num).repeat(b, 1, 1, 1)
        depth_agent_bias = depth_agent_bias1 + depth_agent_bias2

        # second cross agent
        rgb_q_attn = self.softmax((rgb_q * self.scale) @ depth_agent_tokens.transpose(-2, -1) + depth_agent_bias)
        rgb_q_attn = self.rgb_attn_drop(rgb_q_attn)
        rgb_fea = rgb_q_attn @ depth_agent_v
        depth_q_attn = self.softmax((depth_q * self.scale) @ rgb_agent_tokens.transpose(-2, -1) + rgb_agent_bias)
        depth_q_attn = self.depth_attn_drop(depth_q_attn)
        depth_fea = depth_q_attn @ rgb_agent_v

        # 1.rgb
        rgb_fea = rgb_fea.transpose(1, 2).reshape(b, n, c)
        rgb_v = rgb_v.transpose(1, 2).reshape(b, H // self.sr_ratio_rgb, W // self.sr_ratio_rgb, c).permute(0, 3, 1, 2)
        if self.sr_ratio_rgb > 1:
            rgb_v = nn.functional.interpolate(rgb_v, size=(H, W), mode='bilinear')
        rgb_fea = rgb_fea + self.rgb_dwc(rgb_v).permute(0, 2, 3, 1).reshape(b, n, c)
        # 2.depth
        depth_fea = depth_fea.transpose(1, 2).reshape(b, n, c)
        depth_v = depth_v.transpose(1, 2).reshape(b, H // self.sr_ratio_depth, W // self.sr_ratio_depth, c).permute(0, 3, 1, 2)
        if self.sr_ratio_depth > 1:
            depth_v = nn.functional.interpolate(depth_v, size=(H, W), mode='bilinear')
        depth_fea = depth_fea + self.depth_dwc(depth_v).permute(0, 2, 3, 1).reshape(b, n, c)

        # 1.rgb
        rgb_fea = self.rgb_proj(rgb_fea)
        rgb_fea = self.rgb_proj_drop(rgb_fea)
        # 2.depth
        depth_fea = self.depth_proj(depth_fea)
        depth_fea = self.depth_proj_drop(depth_fea)

        return rgb_fea, depth_fea


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# HMSA_coder_1
class Attention_HMSA_decoder_1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 权重初始化函数，对线性层和层归一化层进行不同的初始化
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map=False):
        # 获取输入张量的形状信息
        B, N, C = x.shape
        # 使用线性映射计算查询（q）、键（k）、和值（v）
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 计算注意力分数，并进行缩放
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        # 对注意力分数进行 softmax 操作，并取平均
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N ** .5)
        # 计算注意力分数与位置之间的关系
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        distances = indd ** .5
        distances = distances.to('cuda')

        # 计算位置与注意力分数的加权平均距离
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= N

        if return_map:
            return dist, attn_map
        else:
            return dist

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block_HMSA_decoer_1(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_HMSA_decoder_1(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class MutualSelfBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        # mutual cross agent attention
        self.norm1_rgb_ma = norm_layer(dim)
        self.norm2_depth_ma = norm_layer(dim)
        self.mutualAttn = Mutual_Cross_Agent_Attention(dim, num_patches=(224 // 16) * (224 // 16),
                                                       num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       attn_drop=attn_drop, proj_drop=drop, sr_ratio_rgb=1,
                                                       sr_ratio_depth=1,
                                                       rgb_agent_num=49,
                                                       depth_agent_num=49)  # agent_num=[9, 16, 49, 49]
        self.norm3_rgb_ma = norm_layer(dim)
        self.norm4_depth_ma = norm_layer(dim)
        self.mlp_rgb_ma = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_depth_ma = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # rgb self attention
        self.norm1_rgb_sa = norm_layer(dim)
        self.selfAttn_rgb = Attention_1_convertor_rgb_MHSA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_rgb_sa = norm_layer(dim)
        self.mlp_rgb_sa = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # depth self attention
        self.norm1_depth_sa = norm_layer(dim)
        self.selfAttn_depth = Attention_2_convertor_depth_MHSA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_depth_sa = norm_layer(dim)
        self.mlp_depth_sa = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, rgb_fea, depth_fea):

        # mutual cross agent attention
        rgb_fea_fuse, depth_fea_fuse = self.drop_path(self.mutualAttn(self.norm1_rgb_ma(rgb_fea), self.norm2_depth_ma(depth_fea)))

        rgb_fea = rgb_fea + rgb_fea_fuse
        depth_fea = depth_fea + depth_fea_fuse

        rgb_fea = rgb_fea + self.drop_path(self.mlp_rgb_ma(self.norm3_rgb_ma(rgb_fea)))
        depth_fea = depth_fea + self.drop_path(self.mlp_depth_ma(self.norm4_depth_ma(depth_fea)))

        # rgb self attention
        rgb_fea = rgb_fea + self.drop_path(self.selfAttn_rgb(self.norm1_rgb_sa(rgb_fea)))
        rgb_fea = rgb_fea + self.drop_path(self.mlp_rgb_sa(self.norm2_rgb_sa(rgb_fea)))

        # depth self attention
        depth_fea = depth_fea + self.drop_path(self.selfAttn_depth(self.norm1_depth_sa(depth_fea)))
        depth_fea = depth_fea + self.drop_path(self.mlp_depth_sa(self.norm2_depth_sa(depth_fea)))

        return rgb_fea, depth_fea


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
