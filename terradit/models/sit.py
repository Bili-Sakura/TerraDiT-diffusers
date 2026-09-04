# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import torch.nn.functional as F
from einops import repeat
from .localattn import LocalCrossAttention
from .omega import (InstanceEncoder, InstanceMLP, SpatialGeometryField,
                    GeometryAwareLocalAttention)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0.):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model*2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = zero_module(nn.Linear(d_model, d_model))
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape

        # torch SDPA (correct in fp32/fp16/bf16; no xformers dependency)
        q = self.q_linear(x).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)   # [B,H,Lq,d]
        kv = self.kv_linear(cond).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)   # [B,H,Lk,d]
        v = v.transpose(1, 2)
        attn_mask = None
        if mask is not None:
            # key-padding mask: [B, Lk] (1=keep, 0=pad) -> additive [B,1,1,Lk]
            attn_mask = torch.zeros(B, 1, 1, k.shape[2], dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(mask.reshape(B, 1, 1, -1) == 0, float('-inf'))
        p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p)
        x = x.transpose(1, 2).contiguous().reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

def modulate(x, shift, scale):
    #if condition=='class' or condition=='unconditional':
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    # elif condition=='text':
    #     #return x * (1 + scale) + shift
    #     attn = torch.nn.functional.softmax(torch.einsum('bnd,bkd->bnk', x, scale) / math.sqrt(x.shape[-1]), dim=-1)
    #     return torch.einsum('bnk,bkd->bnd', attn, shift)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################            
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        # freqs = torch.exp(
        #     -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        # ).to(device=t.device)
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class PromptEmbedder(nn.Module):
    def __init__(self, dim, hidden_size):
        super().__init__()
        self.embedder = build_mlp(dim, 2*dim, hidden_size)
    def forward(self, c):
        return self.embedder(c)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, condition='class',
                 point_prompts=False, local_attn=False, omega=True, omega_attn='GALA', **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.local_attn_flag = local_attn
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        if condition=='text':
            self.cross_attn = MultiHeadCrossAttention(hidden_size, 8)
        if point_prompts:
            if local_attn:
                self.local_attn = LocalCrossAttention(hidden_size, 8)
            else:
                self.local_attn = MultiHeadCrossAttention(hidden_size, 8)
        if omega:
            # Only GALA (Geometry-Aware Local Attention) is retained in the public
            # release; the other attention variants were ablation-only.
            if omega_attn == 'GALA':
                self.omega_attn = GeometryAwareLocalAttention(hidden_size, 8)
            else:
                raise NotImplementedError(
                    f"omega_attn={omega_attn!r} is not available in this release (only 'GALA')."
                )
        self.condition=condition
        self.point_prompts=point_prompts
        self.omega=omega

    def forward(self, x, c, point_prompts=None, pos=None, mask=None, t=None, 
                inst_tokens=None, polygon_xy=None, polygon_xy_mask=None,
                polyline_xy=None, polyline_xy_mask=None, bbox_xyxy=None, 
                point_xy=None, format_mask=None, instance_mask=None):
        
        if self.condition=='class' or self.condition=='unconditional':
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(6, dim=-1)
            )
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            if self.omega:
                x = x + self.omega_attn(x=x, inst_tokens=inst_tokens, point_xy=point_xy, polygon_xy=polygon_xy, 
                                        polygon_xy_mask=polygon_xy_mask, polyline_xy=polyline_xy, polyline_xy_mask=polyline_xy_mask, 
                                        bbox_xyxy=bbox_xyxy, format_mask=format_mask, instance_mask=instance_mask)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(t).chunk(6, dim=-1)
            )
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + self.cross_attn(x, c)
            if self.point_prompts:
                if self.local_attn_flag:
                    x = x + self.local_attn(x, point_prompts, pos, mask)
                else:
                    x = x + self.local_attn(x, point_prompts, mask)
            if self.omega:
                x = x + self.omega_attn(x=x, inst_tokens=inst_tokens, point_xy=point_xy, polygon_xy=polygon_xy, 
                                        polygon_xy_mask=polygon_xy_mask, polyline_xy=polyline_xy, polyline_xy_mask=polyline_xy_mask, 
                                        bbox_xyxy=bbox_xyxy, format_mask=format_mask, instance_mask=instance_mask)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        projector_dim=2048,
        condition_type='text',
        condition_dim=768,
        geolocation=False,
        geolocation_dim=32,
        point_prompts=False,
        use_repa=True,
        legacy=True, # whether to use legacy text conditioning (cross-attn with t-emb only)
        local_attn=True,
        omega=True,
        omega_attn='GALA',
        **block_kwargs # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        if condition_type=='text':
            self.y_embedder = PromptEmbedder(condition_dim, hidden_size)
        elif condition_type=='class':
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        if geolocation:
            self.geo_embedder = zero_module(PromptEmbedder(geolocation_dim, hidden_size))
        if point_prompts:
            self.point_pos_encoding = PromptEmbedder(4, hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, condition=condition_type, point_prompts=point_prompts, 
                     local_attn=local_attn, omega=omega, omega_attn=omega_attn, **block_kwargs) for _ in range(depth)
        ])
        if use_repa:
            self.projectors = nn.ModuleList([
                build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
                ])
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        self.condition_type=condition_type
        self.geolocation=geolocation
        self.point_prompts=point_prompts
        self.legacy = legacy
        self.use_repa = use_repa
        self.omega=omega
        self.omega_attn=omega_attn
        if self.omega:
            self.instance_encoder = InstanceEncoder(hidden_size, condition_dim)
        else:
            self.instance_encoder = None

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def forward(self, x, t, y, y_pooled=None, point_prompts=None, pos=None, mask=None, loc_embed=None, return_logvar=False,
                inst_text_embed=None, polygon_xy=None, polygon_xy_mask=None, polyline_xy=None, polyline_xy_mask=None, 
                bbox_xyxy=None, point_xy=None, format_mask=None, instance_mask=None, dropout_probs=None):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape

        # timestep and class embedding
        t_embed = self.t_embedder(t)                   # (N, D)
        if self.condition_type=='unconditional':
            c = t_embed
            if self.geolocation:
                loc_embed = self.geo_embedder(loc_embed)
                c = c + loc_embed
            if self.omega:
                inst_tokens, inst_mask, format_mask_dropped = self.instance_encoder(inst_text_embed, polygon_xy, polygon_xy_mask, 
                                                               polyline_xy, polyline_xy_mask, bbox_xyxy, point_xy, format_mask, 
                                                               instance_mask, dropout_probs=dropout_probs,)
        elif self.condition_type=='class':
            y = self.y_embedder(y, self.training)    # (N, D)
            if self.geolocation:
                loc_embed = self.geo_embedder(loc_embed)
                c = t_embed + y + loc_embed               # (N, D)
            else:
                c = t_embed + y                                # (N, D)
        else:
            #y = self.y_embedder(y)
            #c = torch.cat((t_embed.unsqueeze(1), y), dim=1)
            c = self.y_embedder(y)    # (B, N, D)
            y_pooled = self.y_embedder(y_pooled)
            if self.point_prompts:
                point_prompts = self.y_embedder(point_prompts)
                point_pos_enc = self.point_pos_encoding(torch.cat((torch.sin(2*np.pi*pos/255), torch.cos(2*np.pi*pos/255)), dim=-1))
                point_prompts = point_prompts + point_pos_enc
                # point_prompts = torch.cat(((y_pooled+t_embed).unsqueeze(1), point_prompts), dim=1)
                point_prompts = torch.cat((y_pooled.unsqueeze(1), point_prompts), dim=1)
            if self.omega:
                inst_tokens, inst_mask, format_mask_dropped = self.instance_encoder(inst_text_embed, polygon_xy, polygon_xy_mask, 
                                                               polyline_xy, polyline_xy_mask, bbox_xyxy, point_xy, format_mask, 
                                                               instance_mask, dropout_probs=dropout_probs,)
            if self.geolocation:
                loc_embed = self.geo_embedder(loc_embed)
            #c = t_embed + y

        for i, block in enumerate(self.blocks):
            if self.condition_type=='text':
                if self.geolocation:
                    if self.legacy:
                        if self.point_prompts:
                            x = block(x, c, point_prompts, pos//16, mask, t_embed+loc_embed)
                        else:
                            x = block(x, c, t=t_embed+loc_embed)
                    else:
                        if self.point_prompts:
                            x = block(x, c, point_prompts, pos//16, mask, y_pooled+t_embed+loc_embed)
                        elif self.omega:
                            x = block(x, c, mask=mask, t=y_pooled+t_embed+loc_embed, inst_tokens=inst_tokens, instance_mask=inst_mask,
                                      polygon_xy=polygon_xy, polygon_xy_mask=polygon_xy_mask, polyline_xy=polyline_xy, polyline_xy_mask=polyline_xy_mask, 
                                      bbox_xyxy=bbox_xyxy, point_xy=point_xy, format_mask=format_mask_dropped)                
                        else:
                            x = block(x, c, t=y_pooled+t_embed+loc_embed)
                else:
                    if self.legacy:
                        if self.point_prompts:
                            x = block(x, c, point_prompts, pos//16, mask, t_embed)
                        else:
                            x = block(x, c, t=t_embed)
                    else:
                        if self.point_prompts:
                            x = block(x, c, point_prompts, pos//16, mask, y_pooled+t_embed)
                        if self.omega:
                            x = block(x, c, mask=mask, t=y_pooled+t_embed, inst_tokens=inst_tokens, instance_mask=inst_mask,
                                      polygon_xy=polygon_xy, polygon_xy_mask=polygon_xy_mask, polyline_xy=polyline_xy, polyline_xy_mask=polyline_xy_mask, 
                                      bbox_xyxy=bbox_xyxy, point_xy=point_xy, format_mask=format_mask_dropped)
                        else:
                            x = block(x, c, t=y_pooled+t_embed)
            else:
                if self.omega:
                    x = block(x, c, mask=mask, t=t_embed, inst_tokens=inst_tokens, instance_mask=inst_mask,
                                      polygon_xy=polygon_xy, polygon_xy_mask=polygon_xy_mask, polyline_xy=polyline_xy, polyline_xy_mask=polyline_xy_mask, 
                                      bbox_xyxy=bbox_xyxy, point_xy=point_xy, format_mask=format_mask_dropped)
                else:
                    x = block(x, c)                      # (N, T, D)
            if (i + 1) == self.encoder_depth:
                if self.use_repa:
                    zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
                else:
                    zs = None
        if self.condition_type=='text':
            if self.geolocation:
                x = self.final_layer(x, y_pooled+t_embed+loc_embed)                # (N, T, patch_size ** 2 * out_channels)
            else:
                x = self.final_layer(x, y_pooled+t_embed)                # (N, T, patch_size ** 2 * out_channels)
        else:
            x = self.final_layer(x, c)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x, zs


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}

