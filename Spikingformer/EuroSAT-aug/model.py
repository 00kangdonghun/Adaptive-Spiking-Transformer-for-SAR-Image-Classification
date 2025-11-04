# ======================================================================================
# ✨ Custom Data Augmentation Classes for SAR Dataset ✨
# ======================================================================================
import cv2
import numpy as np
import torch
import random
from PIL import Image

class SpeckleNoise:
    """
    Adds multiplicative speckle noise to the image.
    """
    def __init__(self, variance=0.1):
        self.variance = variance

    def __call__(self, img):
        np_img = np.array(img)
        h, w = np_img.shape[:2]
        
        # Generate noise following a Gamma distribution (mean 1)
        noise = np.random.gamma(1 / self.variance, self.variance, (h, w))
        
        # Reshape noise to match the number of image channels (e.g., Grayscale or RGB)
        if len(np_img.shape) == 3:
            noise = np.stack([noise] * np_img.shape[2], axis=-1)

        # Apply multiplicative noise (clip to 0-255 range)
        noisy_img = np.clip(np_img * noise, 0, 255).astype(np_img.dtype)
        
        return Image.fromarray(noisy_img)

class RandomRotationWithPadding:
    """
    Rotates the image by a random angle and fills the empty space
    at the borders with reflection padding.
    """
    def __init__(self, degrees):
        self.degrees = (-degrees, degrees)

    def __call__(self, img):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        np_img = np.array(img)
        h, w = np_img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform the rotation transformation applying reflection padding (BORDER_REFLECT)
        rotated_img = cv2.warpAffine(np_img, rotation_matrix, (w, h), 
                                     borderMode=cv2.BORDER_REFLECT)
        
        return Image.fromarray(rotated_img)

    
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

__all__ = ['Spikingformer']


# ----------------------------
# MLP block (SNN style)
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., lif_tau=2.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.mlp1_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend='cupy')
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, bias=False)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend='cupy')
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, bias=False)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):  # x: [T,B,C,H,W]
        T, B, C, H, W = x.shape
        x = self.mlp1_lif(x)
        x = self.mlp1_bn(self.mlp1_conv(x.flatten(0, 1))).reshape(T, B, self.c_hidden, H, W).contiguous()

        x = self.mlp2_lif(x)
        x = self.mlp2_bn(self.mlp2_conv(x.flatten(0, 1))).reshape(T, B, C, H, W).contiguous()
        return x


# ----------------------------
# Spiking Self-Attention (SSA)
# ----------------------------
class SpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, scale=0.125, lif_tau=2.0,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = scale

        self.proj_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend='cupy')

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(tau=lif_tau, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):  # x: [T,B,C,H,W]
        T, B, C, H, W = x.shape

        x = self.proj_lif(x)
        x = x.flatten(3)  # [T,B,C,N]
        T, B, C, N = x.shape
        xf = x.flatten(0, 1)  # [T*B, C, N]

        # Q
        q = self.q_lif(self.q_bn(self.q_conv(xf)).reshape(T, B, C, N)).transpose(-1, -2)
        q = q.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        # K
        k = self.k_lif(self.k_bn(self.k_conv(xf)).reshape(T, B, C, N)).transpose(-1, -2)
        k = k.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        # V
        v = self.v_lif(self.v_bn(self.v_conv(xf)).reshape(T, B, C, N)).transpose(-1, -2)
        v = v.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale                          # [T,B,h,N,d]
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        return x


# ----------------------------
# Transformer Block
# ----------------------------
class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., scale=0.125, lif_tau=2.0,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, scale=scale, lif_tau=lif_tau,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, lif_tau=lif_tau)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


# ----------------------------
# Tokenizer (SNN + multi-stage pooling)
# ----------------------------
class SpikingTokenizer(nn.Module):
    def __init__(self, img_size_h=64, img_size_w=64, patch_size=8,
                 in_channels=3, embed_dims=384, stages=None, lif_tau=2.0):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        # auto stages from img_size/patch_size (ex. 128/8=16 -> log2=4 stages)
        target_red_h = img_size_h // patch_size[0]
        assert (target_red_h & (target_red_h - 1)) == 0, "img_size/patch_size must be power of 2"
        auto_stages = int(torch.tensor(target_red_h).log2().item())
        self.stages = auto_stages if stages is None else stages

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)

        def blk(ic, oc):
            return nn.ModuleDict({
                "lif":  MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend='cupy'),
                "pool": nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                "conv": nn.Conv2d(ic, oc, kernel_size=3, stride=1, padding=1, bias=False),
                "bn":   nn.BatchNorm2d(oc),
            })

        c1, c2, c3, c4 = embed_dims // 8, embed_dims // 4, embed_dims // 2, embed_dims
        chs = [c1, c2, c3, c4, embed_dims]
        self.blocks = nn.ModuleList()
        for i in range(self.stages):
            self.blocks.append(blk(chs[i], chs[i + 1]))

    def forward(self, x):  # x: [T,B,C,H,W]
        T, B, C, H, W = x.shape
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, -1, H, W).contiguous()
        for i, blk in enumerate(self.blocks):
            x = blk["lif"](x).flatten(0, 1)
            x = blk["pool"](x)
            x = blk["conv"](x)
            x = blk["bn"](x)
            H, W = H // 2, W // 2
            x = x.reshape(T, B, -1, H, W).contiguous()
        return x, (H, W)


# ----------------------------
# Top-level model
# ----------------------------
class vit_snn(nn.Module):
    def __init__(self,
                 img_size_h=64, img_size_w=64, patch_size=8, in_channels=3, num_classes=10,
                 embed_dims=384, num_heads=6, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=8, sr_ratios=1, T=4, spike_scale=0.125, lif_tau=2.0, stages=None,
                 pretrained_cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths if isinstance(depths, int) else sum(depths)
        self.T = T

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depths)]

        self.patch_embed = SpikingTokenizer(
            img_size_h=img_size_h, img_size_w=img_size_w,
            patch_size=patch_size, in_channels=in_channels,
            embed_dims=embed_dims, stages=stages, lif_tau=lif_tau
        )

        self.block = nn.ModuleList([
            SpikingTransformer(
                dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios,
                scale=spike_scale, lif_tau=lif_tau, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
                norm_layer=norm_layer, sr_ratio=sr_ratios
            ) for j in range(self.depths)
        ])

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, *args, **kwargs):
        # not used (kept for API compatibility)
        return None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):  # x: [T,B,C,H,W]
        x, (H, W) = self.patch_embed(x)
        for blk in self.block:
            x = blk(x)
        return x.flatten(3).mean(3)  # GAP over spatial

    def forward(self, x):  # x: [B,C,H,W]
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)  # [T,B,C,H,W]
        x = self.forward_features(x)                     # [T,B,C]
        x = self.head(x.mean(0))                         # time average
        return x


@register_model
def Spikingformer(pretrained=False, **kwargs):
    model = vit_snn(**kwargs)
    model.default_cfg = _cfg()
    return model


# For quick local test
if __name__ == '__main__':
    from timm.models import create_model
    x = torch.randn(2, 3, 64, 64).cuda()
    model = create_model(
        'Spikingformer',
        img_size_h=64, img_size_w=64,
        patch_size=8, embed_dims=384, num_heads=6, mlp_ratios=4,
        in_channels=3, num_classes=10, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=8, sr_ratios=1,
        T=4, spike_scale=0.125, lif_tau=2.0, stages=None
    ).cuda()
    model.eval()
    y = model(x)
    print(y.shape)
    print('Test Good!')
