import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import math
import logging
from functools import partial
from collections import OrderedDict

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import Block

from util.pos_embed import get_2d_sincos_pos_embed
from util.hog_layer import HOGLayerC

from einops import rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PVTBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, block_size=32,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, **kwargs):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages
        self.img_size = img_size
        self.block_size = block_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            pvtblock = nn.ModuleList([PVTBlock(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", pvtblock)
            setattr(self, f"norm{i + 1}", norm)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pvtblock = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in pvtblock:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x
    
    def mim_block_masking(self, x, mask_ratio, block_size=32):
        batch, channel, height, width = x.shape
        input_size = self.img_size        
        assert height == width, f"Input height and width doesn't match ({height} != {width})."
        
        mask_size = input_size // block_size
        bw_ratio = height // mask_size
        len_keep = int(mask_size**2 * (1 - mask_ratio))
        
        noise = torch.rand(batch, mask_size**2, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        loss_mask = torch.ones([batch, mask_size**2], device=x.device)
        loss_mask[:, :len_keep] = 0
        loss_mask = torch.gather(loss_mask, dim=1, index=ids_restore)
        loss_mask = loss_mask.reshape(batch, 1, mask_size, mask_size).long()

        
        mask = loss_mask.repeat(1, bw_ratio**2, 1, 1)
        mask = mask.reshape(batch, bw_ratio, bw_ratio, mask_size, mask_size).permute(
            0, 3, 1, 4, 2).reshape(batch, 1, height, width)
        
        if self.block_size > 32:
            loss_mask = torch.repeat_interleave(loss_mask, self.block_size//32, dim=2)
            loss_mask = torch.repeat_interleave(loss_mask, self.block_size//32, dim=3)
        
        return mask, loss_mask
    
    def forward(self, x, mask_ratio, mask_token):
        mask, loss_mask = self.mim_block_masking(x, mask_ratio, block_size=self.block_size)
        B, C, H, W = x.shape
        x = x * (1-mask) + (mask) * mask_token.repeat(B, 1, H, W)

        x = self.forward_features(x)

        return x, loss_mask 


class PretrainPVTv2(nn.Module):    
    def __init__(self,
                 img_size=224, decoder_embed_dim=512, decoder_depth=8, block_size=32, output_stride=32,
                 embed_dims=[64, 128, 256, 512], drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=(2,2,18,2), num_heads=(1,2,4,8), sr_ratios=[8,4,2,1], mlp_ratios=[4,4,4,4],
                 norm_pix_loss=False, mim_loss='HOG', **kwargs):
        super().__init__()
        self.img_size = img_size
        self.block_size = block_size
        self.output_stride = output_stride
        self.norm_pix_loss = norm_pix_loss
        self.mim_loss = mim_loss
        self.hog_nbins = kwargs.get('hog_nbins', 9)
        self.hog_pool = kwargs.get('hog_pool', 8)

        decoder_num_heads = int(decoder_embed_dim / 32)
        model_kwargs = dict(
            img_size=img_size, block_size=block_size, embed_dims=embed_dims, norm_layer=norm_layer,
            depths=depths, num_heads=num_heads, sr_ratios=sr_ratios, mlp_ratios=mlp_ratios, **kwargs
        )

        self.encoder = PyramidVisionTransformerV2(drop_path_rate=drop_path_rate, **model_kwargs)
        self.decoder_embed = nn.Linear(
            embed_dims[-1], decoder_embed_dim, bias=True) if decoder_depth > 0 else None
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, 4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        if self.mim_loss == "HOG":
            num_class = (output_stride//self.hog_pool)**2 * self.hog_nbins * 3
        else:
            num_class = output_stride**2 * 3
        self.decoder_pred = nn.Linear(decoder_embed_dim, num_class, bias=True) # encoder to decoder

        self.apply(self._init_weights)
        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1))
        nn.init.normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}
    
    def patchify(self, imgs, p):
        """
        imgs: (N, 3, H, W)
        x: (N, L, output_stride**2 *3)
        """
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->npqchw', x)
        x = x.reshape(shape=(imgs.shape[0], p**2 * 3, h, w))
        return x
    
    def forward_l2_loss(self, imgs, pred, mask):
        B, N, C = pred.shape
        H = W = int(N**0.5)
        
        if self.block_size >= self.output_stride:
            pred = pred.transpose(-1,-2).reshape(B, C, H, W)
            target = self.patchify(imgs, self.output_stride)
        elif self.block_size < self.output_stride:
            pred = pred.reshape(B, H, W, -1, self.output_stride//self.block_size, self.output_stride//self.block_size)
            pred = torch.einsum('nhwcpq->nhpwqc', pred)
            pred = pred.reshape(B, imgs.shape[2]//self.block_size, imgs.shape[3]//self.block_size, -1)
            pred = pred.permute(0, 3, 1, 2)
            target = self.patchify(imgs, self.block_size)
            
        if self.norm_pix_loss:
            mean = target.mean(dim=1, keepdim=True)
            var = target.var(dim=1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        mask = mask.repeat(1, target.shape[1], 1, 1).bool()
        loss = (pred[mask] - target[mask]) ** 2
        loss = loss.mean()
        return loss
    
    def forward_hog_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        B, N, C = pred.shape
        H = W = int(N**0.5)
        mask_size = mask.shape[-1]
        
        hogC = HOGLayerC(nbins=self.hog_nbins, pool=self.hog_pool, norm_pix_loss=self.norm_pix_loss).cuda()
        target = hogC(imgs)

        if mask_size > W:
            target_size, target_channel = target.shape[3], target.shape[1]
            target = target.permute(0, 2, 3, 1).flatten(1, 2)
            assert target_size >= mask_size, "Need larger target size than mask size"
            mask = torch.repeat_interleave(mask, target_size//mask_size, dim=2)
            mask = torch.repeat_interleave(mask, target_size//mask_size, dim=3)
            mask = mask.flatten(1).bool()
            pred = pred.reshape(B, H, W, -1, target_size//H, target_size//W)
            pred = torch.einsum('nhwcpq->nhpwqc', pred)
            pred = pred.reshape(B, target_size**2, target_channel)
        else:
            unfold_size = target.shape[-1] // W
            target = (
                target.permute(0, 2, 3, 1)
                .unfold(1, unfold_size, unfold_size)
                .unfold(2, unfold_size, unfold_size)
                .flatten(1, 2).flatten(2)
            )
            mask = mask.flatten(1).bool()
        
        loss = (pred[mask] - target[mask]) ** 2
        loss = loss.mean()
        
        return loss

    def forward(self, imgs, mask_ratio=0.25, mask_type=None):
        x, mask = self.encoder(imgs, mask_ratio, self.mask_token)

        if self.decoder_blocks:
            x = self.decoder_embed(x)
            for blk in self.decoder_blocks:
                x = blk(x)
        
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        
        if self.mim_loss == 'l2':
            loss = self.forward_l2_loss(imgs, x, mask)
        elif self.mim_loss == 'HOG':
            loss = self.forward_hog_loss(imgs, x, mask)
        else:
            raise NotImplementedError('Illegal mim loss.')
        
        return loss, x, imgs


@register_model
def mim_pvtv2_b2(**kwargs):
    model = PretrainPVTv2(
        embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0., **kwargs)
    return model

@register_model
def mim_pvtv2_b5(**kwargs):
    model = PretrainPVTv2(
        embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
        depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1], qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0., **kwargs)
    return model
