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

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = dim // qk_ratio

        self.q = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr_ratio = sr_ratio
        # Exactly same as PVTv1
        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=True),
                nn.BatchNorm2d(dim, eps=1e-5),
            )

    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            k = self.k(x_).reshape(B, -1, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.k(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + relative_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CMTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        
    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape
        cnn_feat = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)
    

class CMT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[46,92,184,368], stem_channel=16, fc_dim=1280,
                 num_heads=[1,2,4,8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 depths=[2,2,10,2], qk_ratio=1, sr_ratios=[8,4,2,1], dp=0.1, block_size=32, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dims[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.img_size = img_size
        self.block_size = block_size
        self.center_rate = 0.
        
        self.stem_conv1 = nn.Conv2d(3, stem_channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)
        
        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)
        
        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.patch_embed_a = PatchEmbed(
            img_size=img_size//2, patch_size=2, in_chans=stem_channel, embed_dim=embed_dims[0])
        self.patch_embed_b = PatchEmbed(
            img_size=img_size//4, patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed_c = PatchEmbed(
            img_size=img_size//8, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed_d = PatchEmbed(
            img_size=img_size//16, patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.relative_pos_a = nn.Parameter(torch.randn(
            num_heads[0], self.patch_embed_a.num_patches, self.patch_embed_a.num_patches//sr_ratios[0]//sr_ratios[0]))
        self.relative_pos_b = nn.Parameter(torch.randn(
            num_heads[1], self.patch_embed_b.num_patches, self.patch_embed_b.num_patches//sr_ratios[1]//sr_ratios[1]))
        self.relative_pos_c = nn.Parameter(torch.randn(
            num_heads[2], self.patch_embed_c.num_patches, self.patch_embed_c.num_patches//sr_ratios[2]//sr_ratios[2]))
        self.relative_pos_d = nn.Parameter(torch.randn(
            num_heads[3], self.patch_embed_d.num_patches, self.patch_embed_d.num_patches//sr_ratios[3]//sr_ratios[3]))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks_a = nn.ModuleList([
            CMTBlock(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        cur += depths[0]
        self.blocks_b = nn.ModuleList([
            CMTBlock(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        cur += depths[1]
        self.blocks_c = nn.ModuleList([
            CMTBlock(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        cur += depths[2]
        self.blocks_d = nn.ModuleList([
            CMTBlock(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        self.norm = norm_layer(embed_dims[-1])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # def forward_features(self, x, mask_ratio, mask_token):
    def forward_features(self, x):
        B = x.shape[0]
        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)
        
        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)
        
        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)
        
        x, (H, W) = self.patch_embed_a(x)
        for i, blk in enumerate(self.blocks_a):
            x = blk(x, H, W, self.relative_pos_a)
            
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_b(x)
        for i, blk in enumerate(self.blocks_b):
            x = blk(x, H, W, self.relative_pos_b)
            
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_c(x)
        for i, blk in enumerate(self.blocks_c):
            x = blk(x, H, W, self.relative_pos_c)
            
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_d(x)
        for i, blk in enumerate(self.blocks_d):
            x = blk(x, H, W, self.relative_pos_d)
        
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
        x = self.norm(x)

        return x, loss_mask 


class PretrainCMT(nn.Module):    
    def __init__(self,
                 img_size=224, decoder_embed_dim=512, decoder_depth=8, block_size=32, 
                 drop_rate=0., embed_dims=[46,92,184,368],
                 depths=(2,2,18,2), num_heads=(1,2,4,8), sr_ratios=[8,4,2,1],
                 drop_path_rate=0., norm_layer=nn.LayerNorm,
                 norm_pix_loss=False, mim_loss='HOG', **kwargs):
        super().__init__()
        self.img_size = img_size
        self.mim_loss = mim_loss
        self.block_size = block_size
        self.patch_size = 32
        self.norm_pix_loss = norm_pix_loss
        self.hog_nbins = kwargs.get('hog_nbins', 9)
        self.hog_pool = kwargs.get('hog_pool', 8)

        decoder_num_heads = int(decoder_embed_dim / 32)

        model_kwargs = dict(
            img_size=img_size, block_size=block_size, embed_dims=embed_dims, norm_layer=norm_layer,
            depths=depths, num_heads=num_heads, sr_ratios=sr_ratios, **kwargs
        )

        self.encoder = CMT(drop_path_rate=drop_path_rate, **model_kwargs)
        
        self.decoder_embed = nn.Linear(
            embed_dims[-1], decoder_embed_dim, bias=True) if decoder_depth > 0 else None
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, 4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        if self.mim_loss == "HOG":
            num_class = (32//8)**2 * 9 * 3
        else:
            num_class = block_size**2 * 3
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
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->npqchw', x)
        x = x.reshape(shape=(imgs.shape[0], p**2 * 3, h, w))
        return x
    
    def forward_l2_loss(self, imgs, pred, mask):
        B, N, C = pred.shape
        H = W = int(N**0.5)
        pred = pred.transpose(-1,-2).reshape(B, C, H, W)
        
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=1, keepdim=True)
            var = target.var(dim=1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        mask = mask.repeat(1, C, 1, 1).bool()

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
            mask = torch.repeat_interleave(mask, target_size//mask_size, dim=2)
            mask = torch.repeat_interleave(mask, target_size//mask_size, dim=3)
            mask = mask.flatten(1).bool()
            pred = pred.reshape(B, H, W, -1, target_size//H, target_size//W).permute(0, 1, 4, 2, 5, 3).reshape(B, target_size**2, target_channel)
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
            raise NotImplementedError('Undefined MIM loss...')
        
        return loss, x, imgs


@register_model
def mim_cmt_small(**kwargs):
    model = PretrainCMT(
        embed_dims=[64,128,256,512], stem_channel=32, num_heads=[1,2,4,8], depths=[3,3,16,3], sr_ratios=[8,4,2,1],
        qkv_bias=True, norm_layer=None, drop_path_rate=0., **kwargs)
    return model
