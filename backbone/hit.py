# 2021.06.15-Changed for implementation of TNT model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929
The official jax code is released and available at https://github.com/google-research/vision_transformer
Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
from functools import partial
import math

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

from einops import rearrange
import torch.nn.functional as F
# from adaPool import adapool1d
# from SoftPool import soft_pool1d, SoftPool1d

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

def init_skim_predictor(module_list, mean_bias=3.0):
    for module in module_list:
        if not isinstance(module, torch.nn.Linear):
            raise ValueError("only support initialization of linear skim predictor")

        # module.bias.data[1].fill_(5.0)
        # module.bias.data[0].fill_(-5.0)
        # module.weight.data.zero_()
        # module.bias.data[3].normal_(mean=-mean_bias, std=0.02)
        # module.bias.data[2].normal_(mean=-mean_bias, std=0.02)
        module.bias.data[0].normal_(mean=-mean_bias, std=0.02)
        module.bias.data[1].normal_(mean=mean_bias, std=0.02)
        module.weight.data.normal_(mean=0.0, std=0.02)

        module._skim_initialized = True

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'tnt_s_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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


class SE(nn.Module):
    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True) # B, 1, C
        a = self.fc(a)
        x = a * x
        return x


class Attention(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

        self.attention_map = None
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]   # make torchscript happy (cannot use tensor as tuple)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        self.save_attention_map(attn)
        # import ipdb;ipdb.set_trace()

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class ConvPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, bias=True)
        self.norm = nn.LayerNorm(out_channels)
        # self.pool = nn.MaxPool2d(kernel_size=5, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1)

    def forward(self, x):
        """
        x is expected to have shape (B, C, H, W)
        """
        # _assert(x.shape[-2] % 2 == 0, 'BlockAggregation requires even input spatial dims')
        # _assert(x.shape[-1] % 2 == 0, 'BlockAggregation requires even input spatial dims')
        x = self.conv(x)
        # Layer norm done over channel dim only
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.pool(x)
        return x  # (B, C, H//2, W//2)


class Block(nn.Module):
    """ TNT Block
    """
    def __init__(self, outer_dim, inner_dim, outer_num_heads, inner_num_heads, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0):
        super().__init__()
        self.outer_num_heads = outer_num_heads
        self.has_inner = inner_dim > 0
        # if self.has_inner:
        # Inner
        self.inner_norm1 = norm_layer(inner_dim)
        self.inner_attn = Attention(
            inner_dim, inner_dim, num_heads=inner_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.inner_norm2 = norm_layer(inner_dim)
        self.inner_mlp = Mlp(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                out_features=inner_dim, act_layer=act_layer, drop=drop)

        self.proj_norm1 = norm_layer(num_words * inner_dim)
        self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
        self.proj_norm2 = norm_layer(outer_dim)

        # for p in self.parameters():
        #     p.requires_grad=False
        # Outer
        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = Attention(
            outer_dim, outer_dim, num_heads=outer_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(outer_dim)
        self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)
        # SE
        self.se = se
        self.se_layer = None
        if self.se > 0:
            self.se_layer = SE(outer_dim, 0.25)
        
        # self.softpool = SoftPool1d(kernel_size=(16*16), stride=(1))
        self.convpool = ConvPool(inner_dim, outer_dim)

        sizes = [384, 384, 128]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, inner_tokens, outer_tokens, sim_mat, return_attention=False):
        # if self.training:
        #     import ipdb;ipdb.set_trace()
        if self.has_inner:
            # inner_tokens = inner_tokens + self.drop_path(self.inner_attn(self.inner_norm1(inner_tokens))) # B*N, k*k, c
            # inner_tokens = inner_tokens + self.drop_path(self.inner_mlp(self.inner_norm2(inner_tokens))) # B*N, k*k, c
            # B, N, C = outer_tokens.size()
            # tmp_outer_tokens = outer_tokens.squeeze(0).unsqueeze(1).expand(-1, inner_tokens.shape[1], -1)
            # outer_tokens  = (outer_tokens  + torch.max(inner_tokens, dim=1).values) / 2
            # inner_tokens = (inner_tokens + tmp_outer_tokens) / 2
            inner_tokens_res, _ = self.inner_attn(self.inner_norm1(inner_tokens))
            inner_tokens = inner_tokens + self.drop_path(inner_tokens_res) # B*N, k*k, c
            inner_tokens = inner_tokens + self.drop_path(self.inner_mlp(self.inner_norm2(inner_tokens))) # B*N, k*k, c
            B, N, C = outer_tokens.size()
            tmp_outer_tokens = outer_tokens.squeeze(0).unsqueeze(1).expand(-1, inner_tokens.shape[1], -1)

            convpool_inner_tokens = self.convpool(rearrange(inner_tokens, 'b (h w) c -> b c h w', h=8)).squeeze()
            # import ipdb;ipdb.set_trace()
            convpool_inner_tokens_std = (convpool_inner_tokens - convpool_inner_tokens.mean()) / convpool_inner_tokens.std()
            sim_mat_inner = torch.mm(convpool_inner_tokens_std, convpool_inner_tokens_std.T)
            sim_loss = (sim_mat_inner - sim_mat).pow(2)
            # sim_loss = sim_loss.flatten()[:-1].view(h-1,h+1)[:,1:]
            # import ipdb;ipdb.set_trace()

            # barlow-twins loss
            # z1 = self.projector(convpool_inner_tokens)
            # z2 = self.projector(outer_tokens[0, 1:, :].clone().detach())
            # z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
            # z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)
            # cross_corr = torch.matmul(z1_norm.T, z2_norm) / z1.shape[0]
            # on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
            # off_diag = off_diagonal(cross_corr).pow_(2).sum()
            # sim_loss = on_diag + 0.005 * off_diag
            # import ipdb;ipdb.set_trace()

            convpool_inner_tokens_clone = convpool_inner_tokens.clone().detach()
            # convpool_inner_tokens = convpool_inner_tokens.detach()
            # convpool_inner_tokens = inner_tokens.mean(1)
            outer_tokens[:, 1:, :]  = (outer_tokens[:, 1:, :]  + convpool_inner_tokens_clone) / 2
            # outer_tokens[:, 1:, :]  = (outer_tokens[:, 1:, :]  + inner_tokens.mean(1)) / 2
            # outer_tokens[:, 1:, :]  = convpool_inner_tokens
            # outer_tokens[:, 1:, :]  = outer_tokens[:, 1:, :]
            # import ipdb;ipdb.set_trace()
            # outer_tokens  = (outer_tokens  + torch.mean(inner_tokens, dim=1)) / 2
            # inner_tokens = (inner_tokens + tmp_outer_tokens[1:, :, :]) / 2
        if self.se > 0:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            tmp_ = self.outer_mlp(self.outer_norm2(outer_tokens))
            outer_tokens = outer_tokens + self.drop_path(tmp_ + self.se_layer(tmp_))
        else:
            outer_tokens_res, attn = self.outer_attn(self.outer_norm1(outer_tokens))
            outer_tokens = outer_tokens + self.drop_path(outer_tokens_res)
            outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))
        # if return_attention:
        return inner_tokens, outer_tokens, attn, sim_loss
        # return inner_tokens, outer_tokens


class PatchEmbed(nn.Module):
    """ Image to Visual Word Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, outer_dim=768, inner_dim=24, inner_stride=4):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.inner_dim = inner_dim
        self.num_words = math.ceil(patch_size[0] / inner_stride) * math.ceil(patch_size[1] / inner_stride)
        
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv2d(in_chans, inner_dim, kernel_size=7, padding=3, stride=inner_stride)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.unfold(x) # B, Ck2, N
        x = x.transpose(1, 2).reshape(B * self.num_patches, C, *self.patch_size) # B*N, C, 16, 16
        x = self.proj(x) # B*N, C, 8, 8
        x = x.reshape(B * self.num_patches, self.inner_dim, -1).transpose(1, 2) # B*N, 8*8, C
        return x


class HIT(nn.Module):
    """ TNT (Transformer in Transformer) for computer vision
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2, outer_dim=384, inner_dim=384,
                 depth=2, outer_num_heads=2, inner_num_heads=2, mlp_ratio=2., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, inner_stride=4, se=0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.outer_dim = outer_dim  # num_features for consistency with other models

        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, outer_dim=outer_dim,
        #     inner_dim=inner_dim, inner_stride=inner_stride)
        # # self.num_patches = num_patches = self.patch_embed.num_patches
        # num_words = self.patch_embed.num_words
        # num_words = 16 * 16
        num_words = 8 * 8
        
        self.proj_norm1 = norm_layer(num_words * inner_dim)
        self.proj = nn.Linear(num_words * inner_dim, outer_dim)
        self.proj_norm2 = norm_layer(outer_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, outer_dim))
        # self.outer_tokens = nn.Parameter(torch.zeros(1, num_patches, outer_dim), requires_grad=False)
        # self.outer_pos = nn.Parameter(torch.zeros(1, num_patches + 1, outer_dim))
        # self.inner_pos = nn.Parameter(torch.zeros(1, num_words, inner_dim))
        # self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        vanilla_idxs = []
        blocks = []
        for i in range(depth):
            if i in vanilla_idxs:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=-1, outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
            else:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=inner_dim, outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(outer_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(outer_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(outer_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.outer_pos, std=.02)
        # trunc_normal_(self.inner_pos, std=.02)
        self.apply(self._init_weights)


        self.skimpre = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

        self.fc1 = nn.Sequential(nn.Linear(768, inner_dim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(768, inner_dim), nn.ReLU())

        init_skim_predictor([self.skimpre[-1]])

        self.cls_token_inner = nn.Parameter(torch.zeros(1, 1, inner_dim))
        self.cls_token_outer = nn.Parameter(torch.zeros(1, 1, outer_dim))

        self.convpool = ConvPool(inner_dim, outer_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.outer_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features_bak(self, x):
        B = x.shape[0]
        inner_tokens = self.patch_embed(x) + self.inner_pos # B*N, 8*8, C
        
        outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, self.num_patches, -1))))        
        outer_tokens = torch.cat((self.cls_token.expand(B, -1, -1), outer_tokens), dim=1)
        
        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)

        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens)

        outer_tokens = self.norm(outer_tokens)
        return outer_tokens[:, 0]
    
    def forward_features(self, x, x2):
        B = x.shape[0]
        B2 = 1
        x3 = (x2 - x2.mean()) / x2.std()
        # x3 = F.normalize(x2, p=2, dim=1)
        sim_mat = torch.mm(x3, x3.T)
        # inner_tokens = self.patch_embed(x) + self.inner_pos # B*N, 8*8, C
        inner_tokens = x
        # import ipdb;ipdb.set_trace()
        # inner_tokens = torch.cat((self.cls_token_inner.expand(B, -1, -1), x), dim=1)
        
        # outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, self.num_patches, -1))))   
        outer_tokens = x2.unsqueeze(0)
        outer_tokens = torch.cat((self.cls_token.expand(1, -1, -1), outer_tokens), dim=1)
        # outer_tokens  = (outer_tokens  + self.convpool(rearrange(inner_tokens, 'b (h w) c -> b c h w', h=16)).squeeze()) / 2

        # outer_tokens = torch.cat((self.cls_token_outer.expand(B2, -1, -1), outer_tokens), dim=1)
        
        # outer_tokens = self.pos_drop(outer_tokens)
        # import ipdb;ipdb.set_trace()
        sim_loss = 0

        all_layer_attentions = []
        for blk in self.blocks:
            inner_tokens, outer_tokens, attn, loss = blk(inner_tokens, outer_tokens, sim_mat)
            # import ipdb;ipdb.set_trace()
            attn_heads = blk.outer_attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
            sim_loss = sim_loss + loss

        # import ipdb;ipdb.set_trace()
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=0)
        outer_tokens = self.norm(outer_tokens)
        # import ipdb;ipdb.set_trace()
        # return outer_tokens[:, 0]
        # return torch.max(outer_tokens, dim=1).values, attn[:, :, 0, 1:]
        # return outer_tokens[:, 0, :], attn[:, :, 0, 1:], sim_loss
        return outer_tokens[:, 0, :], rollout[:, 0, 1:], sim_loss

    # def forward_mask(self, x, x2):
    #     # import ipdb;ipdb.set_trace()
    #     x2_mask = self.skimpre(x2)
    #     gumbel_mask = nn.functional.gumbel_softmax(x2_mask, hard=True, tau=0.1)
    #     threshold = 0.2
    #     # import ipdb;ipdb.set_trace()
    #     # threshold = torch.sigmoid(torch.sum(x2_mask[:, 1] - x2_mask[:, 0]) / gumbel_mask.shape[0])
    #     skim_loss = (threshold - ((torch.sum(gumbel_mask[:, 0])) / gumbel_mask.shape[0]))**2
    #     # skim_loss = 0

    #     if sum(gumbel_mask[:, 0] == 1) > 0:
    #         x = x[gumbel_mask[:, 0] == 1, :, :, :]
    #         x2 = x2[gumbel_mask[:, 0] == 1, :]
    #     else:
    #         x = x
    #         x2 = x2

    #     x = rearrange(x, 'B H W C -> B (H W) C')
    #     x = self.fc1(x)
    #     x2 = self.fc2(x2)
    #     self.num_patches = x.shape[0]

    #     x = self.forward_features(x, x2)
    #     logits = self.head(x)
    #     Y_hat = torch.argmax(logits)
    #     Y_prob = F.softmax(logits)

    #     return logits, Y_hat, Y_prob, skim_loss
        
    def forward(self, input, returnt='out', inst_feat=False):
        
        # if returnt == 'inst_feat':
        x = input[0]
        x2 = input[1]
        # import ipdb;ipdb.set_trace()

        x = rearrange(x, 'B H W C -> B (H W) C')
        x = self.fc1(x)
        x2 = self.fc2(x2)
        self.num_patches = x.shape[0]

        x, region_attn, sim_loss = self.forward_features(x, x2)
        # region_attn = region_attn.mean(1)
        if returnt == 'features':
            return x
        logits = self.head(x)
        Y_hat = torch.argmax(logits)
        # import ipdb;ipdb.set_trace()
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat, region_attn, sim_loss



def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict
