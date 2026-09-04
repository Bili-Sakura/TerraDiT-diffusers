import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import repeat

def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module

def softmax(x, dim=-1):
    x_exp = torch.exp(x)
    return x_exp / (torch.sum(x_exp, dim=dim, keepdim=True) + 1e-6)


class RBF(nn.Module):
    def __init__(self, d_model, num_heads):
        super(RBF, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_linear = nn.Linear(d_model, d_model*3)
        self.proj = nn.Linear(d_model, 2)

    def forward(self, x, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape

        qkv = self.qkv_linear(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = torch.cat([torch.zeros(B, 1, device=x.device), mask], dim=1)  # for y_pooled
            mask = repeat(mask, 'b k -> b h n k', h=self.num_heads, n=N)
            attn = attn.masked_fill(mask.bool(), float('-inf'))

        attn = softmax(attn, dim=-1)

        x = torch.einsum('bhqk,bhkd->bhqd', attn, v)

        x = x.view(B, -1, C)[:, 1:]  # remove y_pooled
        x = torch.nn.functional.softplus(self.proj(x))
        sigma_x, sigma_y = x.chunk(2, dim=-1)
        sigma_x = torch.clamp(sigma_x, min=1e-2, max=10.0)
        sigma_y = torch.clamp(sigma_y, min=1e-2, max=10.0)
        return sigma_x, sigma_y


class LocalCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(LocalCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rbf = RBF(d_model, num_heads)

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model*2)
        self.proj = zero_module(nn.Linear(d_model, d_model))

    def forward(self, x, cond, pos, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape
        Bc, Nc, Cc = cond.shape

        sigma_x, sigma_y = self.rbf(cond, mask)

        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond[:, 1:]).view(Bc, Nc-1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        q = q.permute(0, 2, 1, 3)   # (B, num_heads, N, head_dim)
        k = k.permute(0, 2, 1, 3)   # (B, num_heads, N, head_dim)
        v = v.permute(0, 2, 1, 3)   # (B, num_heads, N, head_dim)
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(self.head_dim)
        x_pos = torch.arange(int(math.sqrt(N)), device=x.device)
        y_pos = torch.arange(int(math.sqrt(N)), device=x.device)
        X1, X2 = torch.meshgrid(x_pos, y_pos, indexing="ij")
        X1 = repeat(X1.flatten(), 'n -> b n', b=B)
        X2 = repeat(X2.flatten(), 'n -> b n', b=B)
        
        dx2 = -0.5 * torch.einsum('bpn,bnd->bpn', (X1.unsqueeze(-1) - pos[..., 0].unsqueeze(1))**2 , 1 / sigma_x**2 + 1e-5)
        dy2 = -0.5 * torch.einsum('bpn,bnd->bpn', (X2.unsqueeze(-1) - pos[..., 1].unsqueeze(1))**2 , 1 / sigma_y**2 + 1e-5)
        pos_bias = torch.exp(dx2 + dy2)

        if mask is not None:
            mask = repeat(mask, 'b k -> b h n k', h=self.num_heads, n=N)
            attn = attn.masked_fill(mask.bool(), float('-inf'))
        
        attn = softmax(attn, dim=-1)
        attn = torch.einsum('bhnk,bnk->bhnk', attn, pos_bias)
        x = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(B, -1, C)
        x = self.proj(x)

        return x

if __name__ == "__main__":
    B, N, C = 2, 256, 64
    x = torch.randn(B, N, C).cuda()
    cond = torch.randn(B, 11, C).cuda()
    pos = torch.randint(0, 16, (B, 10, 2)).cuda()
    m = torch.zeros(B, 10).cuda()
    attn = LocalCrossAttention(d_model=C, num_heads=8).cuda()
    out = attn(x, cond, pos, m)
    print(out.shape)