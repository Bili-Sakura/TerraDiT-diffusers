import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import repeat

#################################################################################
#                                    Helpers                                    #
################################################################################# 

class FourierEmbedder():
    # https://github.com/frank-xwang/InstanceDiffusion/blob/main/ldm/modules/diffusionmodules/util.py#L12-L26
    def __init__(self, num_freqs, temperature=100):

        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )  

    # @ torch.no_grad()
    # def __call__(self, x, cat_dim=-1):
    #     "x: arbitrary shape of tensor. dim: cat dim"
    #     out = []
    #     for freq in self.freq_bands:
    #         out.append( torch.sin( freq*x ) )
    #         out.append( torch.cos( freq*x ) )
    #     return torch.cat(out, cat_dim)
    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        # FIX: Ensure freq_bands are on the same device as the input tensor x
        freqs = self.freq_bands.to(x.device, dtype=x.dtype)
        
        out = []
        for freq in freqs:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

#################################################################################
#                           Core Omega Conditioning Logic                       #
################################################################################# 

class InstanceEncoder(nn.Module):
    """
    Encode polygon, polyline, bounding box, points into unified representation
    """
    def __init__(self, hidden_dim, text_dim, 
                 polygon_max_points = 64, # or 128
                 polyline_max_points = 64,
                 fourier_polygon_num_freq = 16,
                 fourier_polyline_num_freq = 16,
                 fourier_bbox_num_freq = 8,
                 fourier_point_num_freq = 8,
                 tile_size=256                 
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.polygon_max_points = polygon_max_points
        self.polyline_max_points = polyline_max_points
        
        # self.dropout_polygon = 0.2
        # self.dropout_polyline = 0.2
        # self.dropout_bbox = 0.1 # old 0.2
        
        self.fourier_polygon = FourierEmbedder(num_freqs=fourier_polygon_num_freq)
        self.fourier_polyline = FourierEmbedder(num_freqs=fourier_polyline_num_freq)
        self.fourier_bbox = FourierEmbedder(num_freqs=fourier_bbox_num_freq)
        self.fourier_point = FourierEmbedder(num_freqs=fourier_point_num_freq)
        
        assert hidden_dim % 8 == 0, "hidden_dim must be divisible by 8"
        chunk = hidden_dim // 8
        
        self.polygon_mlp = InstanceMLP(in_dim=text_dim+(polygon_max_points*fourier_polygon_num_freq*4), out_dim=3*chunk)
        self.polyline_mlp = InstanceMLP(in_dim=text_dim+(polyline_max_points*fourier_polyline_num_freq*4), out_dim=3*chunk)
        self.bbox_mlp = InstanceMLP(in_dim=text_dim+(fourier_bbox_num_freq*8), out_dim=1*chunk)
        self.point_mlp = InstanceMLP(in_dim=text_dim+(fourier_point_num_freq*4), out_dim=1*chunk)
        
        self.null_polygon = nn.Parameter(torch.zeros(1, 1, polygon_max_points, fourier_polygon_num_freq*4))
        self.null_polyline = nn.Parameter(torch.zeros(1, 1, polyline_max_points, fourier_polyline_num_freq*4))
        self.null_bbox = nn.Parameter(torch.zeros(1, 1, fourier_bbox_num_freq*8))
        self.null_point = nn.Parameter(torch.zeros(1, 1, fourier_point_num_freq*4))

        self.tile_size = tile_size
        
    def _apply_format_dropout(self, format_mask, dropout_probs=None):
        if dropout_probs is None:
            return format_mask  # default behavior
        
        B, N, F = format_mask.shape
        device = format_mask.device

        keep_probs = torch.tensor([
            1.0 - dropout_probs[0], # poly
            1.0 - dropout_probs[1], # line
            1.0 - dropout_probs[2], # box
            1.0                     # point
        ], device=device).view(1, 1, F)
        
        keep_mask = torch.bernoulli(keep_probs.expand_as(format_mask))
        new_mask = format_mask * keep_mask
        poly_kept = new_mask[..., 0] > 0.5
        line_kept = new_mask[..., 1] > 0.5
        box_exists = format_mask[..., 2] > 0.5
        need_box = (poly_kept | line_kept) & box_exists 
        new_mask[..., 2] = torch.where(need_box, torch.ones_like(new_mask[..., 2]), new_mask[..., 2])
        return new_mask
        
    def forward(self, inst_text_embed,
                polygon_xy, polygon_xy_mask, # mask for real vs. pad coords
                polyline_xy, polyline_xy_mask,
                bbox_xyxy, point_xy,
                format_mask, # formats available for instance
                instance_mask, # real vs. pad instances in image 
                dropout_probs=None,
                ):
        assert polygon_xy.shape[2] == self.polygon_max_points, f"polygon maximum points must be {self.polygon_max_points}"
        assert polyline_xy.shape[2] == self.polyline_max_points, f"polyline maximum points must be {self.polyline_max_points}"
        format_mask = self._apply_format_dropout(format_mask, dropout_probs=dropout_probs)
        B, N, _ = inst_text_embed.shape
        
        # Split by format
        polygon_mask = format_mask[..., 0]
        polyline_mask = format_mask[..., 1]
        bbox_mask = format_mask[..., 2]
        point_mask = format_mask[..., 3]

        # Encode polygon format
        polygon_fourier_per_point = self.fourier_polygon(polygon_xy / self.tile_size) # normalize for stability
        
        format_polygon = polygon_mask.unsqueeze(-1).unsqueeze(-1)
        point_mask_polygon = polygon_xy_mask.unsqueeze(-1)
        # s_polygon = (format_polygon * point_mask_polygon).float()
        s_polygon = (format_polygon * point_mask_polygon).to(inst_text_embed.dtype)

        null_polygon = self.null_polygon
        polygon_fourier_filled = s_polygon * polygon_fourier_per_point + (1.0 - s_polygon) * null_polygon
        polygon_fourier_flat = polygon_fourier_filled.reshape(B, N, polygon_fourier_filled.shape[2] * polygon_fourier_filled.shape[3])

        polygon_input = torch.cat([inst_text_embed, polygon_fourier_flat], dim=-1)
        polygon_feat = self.polygon_mlp(polygon_input)
        
        # Encode polyline format
        polyline_fourier_per_point = self.fourier_polyline(polyline_xy / self.tile_size)

        format_polyline = polyline_mask.unsqueeze(-1).unsqueeze(-1)
        point_mask_polyline = polyline_xy_mask.unsqueeze(-1) 
        # s_polyline = (format_polyline * point_mask_polyline).float()
        s_polyline = (format_polyline * point_mask_polyline).to(inst_text_embed.dtype)

        null_polyline = self.null_polyline
        polyline_fourier_filled = s_polyline * polyline_fourier_per_point + (1.0 - s_polyline) * null_polyline
        polyline_fourier_flat = polyline_fourier_filled.reshape(B, N, polyline_fourier_filled.shape[2] * polyline_fourier_filled.shape[3])

        polyline_input = torch.cat([inst_text_embed, polyline_fourier_flat], dim=-1)
        polyline_feat = self.polyline_mlp(polyline_input)

        # Encode bounding box format
        bbox_fourier = self.fourier_bbox(bbox_xyxy / self.tile_size)
        # bbox_mask_exp = bbox_mask.unsqueeze(-1)
        bbox_mask_exp = bbox_mask.unsqueeze(-1).to(inst_text_embed.dtype)
        null_bbox = self.null_bbox.expand(B, N, -1)
        bbox_fourier_final = bbox_mask_exp * bbox_fourier + (1.0 - bbox_mask_exp) * null_bbox
  
        bbox_input = torch.cat([inst_text_embed, bbox_fourier_final], dim=-1)
        bbox_feat = self.bbox_mlp(bbox_input)
        
        # Encode point format
        point_fourier = self.fourier_point(point_xy / self.tile_size)
        # point_mask_exp = point_mask.unsqueeze(-1)
        point_mask_exp = point_mask.unsqueeze(-1).to(inst_text_embed.dtype)
        null_point = self.null_point.expand(B, N, -1)
        point_fourier_final = point_mask_exp * point_fourier + (1.0 - point_mask_exp) * null_point
        
        point_input = torch.cat([inst_text_embed, point_fourier_final], dim=-1)
        point_feat  = self.point_mlp(point_input)

        # concat
        inst_embed = torch.cat([polygon_feat, polyline_feat, bbox_feat, point_feat], dim=-1,) # [B, N, hidden_dim]

        return inst_embed, instance_mask, format_mask
    
class InstanceMLP(nn.Module):
    """ Embed each instance format """
    def __init__(self, in_dim, out_dim, mid_dim = 3072):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)
    
class MetaRBFPlus(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_linear = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, 3)

    def forward(self, x, mask=None):
        """
        x   : [B, N_inst, C] instance tokens
        mask: [B, N_inst] (1 = real, 0 = pad) or None
        """
        B, N, C = x.shape
        qkv = self.qkv_linear(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            pad = (mask <= 0.5).unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad, float("-inf"))
            all_masked = (mask <= 0.5).all(dim=-1)  # [B]
            if all_masked.any():
                attn[all_masked] = 0.0
        attn = F.softmax(attn, dim=-1)
        x_ctx = torch.einsum("bhqk,bhkd->bhqd", attn, v) 
        x_ctx = x_ctx.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.proj(x_ctx)
        sigma_x = F.softplus(out[..., 0:1])
        sigma_y = F.softplus(out[..., 1:2])
        theta   = out[..., 2:3]
        sigma_x = torch.clamp(sigma_x, min=1e-2, max=128.0) # in pixel space
        sigma_y = torch.clamp(sigma_y, min=1e-2, max=128.0)
        return sigma_x, sigma_y, theta
    
class SpatialGeometryField(nn.Module):
    def __init__(self, init_alpha_poly=1.0, init_alpha_box=1.0, init_alpha_line=1.0, h_type="square", tile_size=256):
        super().__init__()
        self.alpha_poly = nn.Parameter(torch.tensor(init_alpha_poly))
        self.alpha_box  = nn.Parameter(torch.tensor(init_alpha_box))
        self.alpha_line = nn.Parameter(torch.tensor(init_alpha_line))
        self.h_type = h_type
        self.tile_size = float(tile_size)
        
        # cache grid to avoid rebuilding every forward
        self.register_buffer("_cached_grid", None, persistent=False)
        self._cached_hw = None
        
    def _h(self, d):
        if self.h_type == "square":
            return d * d
        elif self.h_type == "abs":
            return d.abs()
        else:
            raise ValueError(f"Unknown h_type: {self.h_type}")
    
    def _make_grid(self, H, W, device):
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        x = (xs + 0.5) * (self.tile_size / W)
        y = (ys + 0.5) * (self.tile_size / H)
        grid = torch.stack([x, y], dim=-1)  # [H, W, 2], (x,y)
        return grid

    def _get_grid(self, H, W, device):
        if (
            self._cached_grid is not None
            and self._cached_hw == (H, W)
            and self._cached_grid.device == device
        ):
            return self._cached_grid
        grid = self._make_grid(H, W, device)
        self._cached_grid = grid
        self._cached_hw = (H, W)
        # return grid
        return self._make_grid(H, W, device)
    
    def _box_sdf(self, grid, bbox_xyxy_flat):
        H, W, _ = grid.shape
        M = bbox_xyxy_flat.shape[0]
        x = grid[..., 0]
        y = grid[..., 1]
        x = x.view(1, 1, H, W)
        y = y.view(1, 1, H, W)
        x_min = bbox_xyxy_flat[:, 0].view(M, 1, 1, 1)
        y_min = bbox_xyxy_flat[:, 1].view(M, 1, 1, 1)
        x_max = bbox_xyxy_flat[:, 2].view(M, 1, 1, 1)
        y_max = bbox_xyxy_flat[:, 3].view(M, 1, 1, 1)
        dx_min = x - x_min
        dx_max = x_max - x
        dy_min = y - y_min
        dy_max = y_max - y
        inside = (dx_min >= 0) & (dx_max >= 0) & (dy_min >= 0) & (dy_max >= 0)
        dx_out = torch.where(x < x_min, x_min - x,
                             torch.where(x > x_max, x - x_max, torch.zeros_like(x)))
        dy_out = torch.where(y < y_min, y_min - y,
                             torch.where(y > y_max, y - y_max, torch.zeros_like(y)))
        dist_outside = torch.sqrt(dx_out * dx_out + dy_out * dy_out)
        dist_inside = -torch.minimum(
            torch.minimum(dx_min, dx_max),
            torch.minimum(dy_min, dy_max),
        )
        sdf = torch.where(inside, dist_inside, dist_outside)
        return sdf[:, 0]
    
    def _polyline_dist(self, grid, verts_flat, verts_mask_flat):
        M, P, _ = verts_flat.shape
        H, W, _ = grid.shape
        Q = H * W
        num_valid = verts_mask_flat.sum(dim=-1)  # [M]
        v0 = verts_flat[:, :-1, :] # segments
        v1 = verts_flat[:, 1:, :]
        seg_mask = verts_mask_flat[:, :-1] & verts_mask_flat[:, 1:]
        Pts = grid.view(Q, 2)
        p_exp = Pts.unsqueeze(0).unsqueeze(2)
        p0 = v0.unsqueeze(1)
        p1 = v1.unsqueeze(1)
        seg = p1 - p0
        seg_norm2 = (seg * seg).sum(-1) + 1e-8
        t = ((p_exp - p0) * seg).sum(-1) / seg_norm2
        t = t.clamp(0.0, 1.0)
        proj = p0 + t.unsqueeze(-1) * seg
        diff = p_exp - proj
        dist = torch.sqrt((diff * diff).sum(-1))
        seg_mask_exp = seg_mask.unsqueeze(1)
        dist = dist.masked_fill(~seg_mask_exp, float("inf"))
        dist_min, _ = dist.min(dim=-1)
        few_mask = num_valid < 2
        if few_mask.any():
            dist_min[few_mask] = 1.0
        dist_map = dist_min.view(M, H, W)
        return dist_map
    
    def _polygon_sdf(self, grid, verts_flat, verts_mask_flat):
        M, P, _ = verts_flat.shape
        H, W, _ = grid.shape
        Q = H * W
        num_valid = verts_mask_flat.sum(dim=-1)
        dist = self._polyline_dist(grid, verts_flat, verts_mask_flat)
        Pts = grid.view(Q, 2)
        px = Pts[:, 0].view(1, Q, 1)
        py = Pts[:, 1].view(1, Q, 1)
        v0 = verts_flat[:, :-1, :]
        v1 = verts_flat[:, 1:, :]
        seg_mask = verts_mask_flat[:, :-1] & verts_mask_flat[:, 1:]
        x0 = v0[..., 0].unsqueeze(1)
        y0 = v0[..., 1].unsqueeze(1)
        x1 = v1[..., 0].unsqueeze(1)
        y1 = v1[..., 1].unsqueeze(1)
        cond_straddle = ((y0 > py) != (y1 > py)) & seg_mask.unsqueeze(1)
        denom = (y1 - y0)
        denom = torch.where(denom == 0, torch.full_like(denom, 1e-8), denom)
        x_int = x0 + (x1 - x0) * (py - y0) / denom
        cond_right = px < x_int
        crossings = cond_straddle & cond_right
        inside_flat = (crossings.sum(dim=-1) % 2 == 1)
        inside = inside_flat.view(M, H, W)
        sdf = torch.where(inside, -dist, dist)
        few_poly = num_valid < 3
        if few_poly.any():
            sdf[few_poly] = 1.0
        return sdf
        
    def forward(self, polygon_xy, polygon_xy_mask,
                polyline_xy, polyline_xy_mask, 
                bbox_xyxy, format_mask, instance_mask, H=16, W=16):
        device = polygon_xy.device
        B, N, P, _ = polygon_xy.shape
        M = B * N
        alpha_poly = F.softplus(self.alpha_poly)
        alpha_box  = F.softplus(self.alpha_box)
        alpha_line = F.softplus(self.alpha_line)
        grid = self._get_grid(H, W, device=device) # pixel/token mapping grid
        
        # flatten for batched calculation
        poly_xy_flat = polygon_xy.view(M, P, 2)
        poly_mask_flat = polygon_xy_mask.view(M, P)
        line_xy_flat = polyline_xy.view(M, P, 2)
        line_mask_flat = polyline_xy_mask.view(M, P)
        bbox_flat = bbox_xyxy.view(M, 4)
        fmt_flat = format_mask.view(M, 4)
        inst_mask_flat = instance_mask.view(M)

        has_poly = fmt_flat[:, 0] > 0.5
        has_line = fmt_flat[:, 1] > 0.5
        has_box  = fmt_flat[:, 2] > 0.5
        has_highres = has_poly | has_line
        use_box = has_box & ~has_highres # only use box if no poly/line
        
        S_poly = torch.ones((M, H, W), device=device)
        S_line = torch.ones((M, H, W), device=device)
        S_box  = torch.ones((M, H, W), device=device)

        # polygon field
        sdf_poly = self._polygon_sdf(grid, poly_xy_flat, poly_mask_flat)
        S_poly_raw = torch.sigmoid(-alpha_poly * sdf_poly)
        poly_mask = has_poly.float().view(M, 1, 1)
        S_poly = 1.0 + poly_mask * (S_poly_raw - 1.0) # ensures alpha in the computation graph
        
        # polyline field
        dist_line = self._polyline_dist(grid, line_xy_flat, line_mask_flat)
        line_scale = self.tile_size / 8.0
        d_norm = dist_line / (line_scale + 1e-6)
        h_line = self._h(d_norm)
        S_line_raw = torch.exp(-alpha_line * h_line)
        line_mask = has_line.float().view(M, 1, 1)
        S_line = 1.0 + line_mask * (S_line_raw - 1.0)
        
        # box field
        sdf_box = self._box_sdf(grid, bbox_flat)
        S_box_raw = torch.sigmoid(-alpha_box * sdf_box)
        box_mask = use_box.float().view(M, 1, 1)
        S_box = 1.0 + box_mask * (S_box_raw - 1.0)

        # combine formats
        S_all_flat = S_poly * S_line * S_box

        if inst_mask_flat is not None:
            inactive = inst_mask_flat <= 0.5
            if inactive.any():
                S_all_flat[inactive] = 1.0
        S_all = S_all_flat.view(B, N, H, W)
        return S_all
    
class GeometryAwareLocalAttention(nn.Module):
    def __init__(self, d_model, num_heads, tile_size=256, init_alpha_poly=1.0,
                 init_alpha_box=1.0, init_alpha_line=1.0, h_type="square"):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rbf = MetaRBFPlus(d_model, num_heads)
        self.geom_field = SpatialGeometryField(init_alpha_poly=init_alpha_poly, init_alpha_box=init_alpha_box,
                                               init_alpha_line=init_alpha_line, h_type=h_type, tile_size=tile_size)
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.proj = zero_module(nn.Linear(d_model, d_model))
        self.tile_size = float(tile_size)
        
    def forward(self, x, inst_tokens, point_xy, polygon_xy, polygon_xy_mask,
                polyline_xy, polyline_xy_mask, bbox_xyxy, format_mask, instance_mask):
        B, N, C = x.shape
        _, N_inst, _ = inst_tokens.shape
        device = x.device

        # infer H, W
        side = int(math.sqrt(N))
        assert side * side == N, f"N={N} is not a perfect square"
        H = W = side

        # rbf kernel
        sigma_x, sigma_y, theta = self.rbf(inst_tokens, mask=instance_mask)
        grid = self.geom_field._get_grid(H, W, device=device)
        coords = grid.view(-1, 2)
        gx = coords[:, 0].unsqueeze(0)
        gy = coords[:, 1].unsqueeze(0)
        px = point_xy[..., 0]
        py = point_xy[..., 1]
        dx = gx.unsqueeze(-1) - px.unsqueeze(1)
        dy = gy.unsqueeze(-1) - py.unsqueeze(1)
        c = torch.cos(theta).transpose(1, 2)
        s = torch.sin(theta).transpose(1, 2)
        s1 = sigma_x.transpose(1, 2)
        s2 = sigma_y.transpose(1, 2)
        d1 = c * dx + s * dy
        d2 = -s * dx + c * dy
        K_pt = torch.exp(-0.5 * ((d1 / (s1 + 1e-6)) ** 2 + (d2 / (s2 + 1e-6)) ** 2))
        if instance_mask is not None:
            active = (instance_mask > 0.5).float()
            K_pt = K_pt * active.unsqueeze(1)

        # spatial geometry field
        S_all = self.geom_field(polygon_xy, polygon_xy_mask, polyline_xy, polyline_xy_mask,
                                bbox_xyxy, format_mask, instance_mask, H=H, W=W)
        S_all = S_all.view(B, N_inst, H * W).transpose(1, 2)
        K_hat = K_pt * S_all
        K_sum = K_hat.sum(dim=1, keepdim=True)
        K_final = K_hat / (K_sum + 1e-6)
        degenerate = (K_sum.squeeze(1) <= 1e-8) # fallback to uniform if degenerate
        if degenerate.any():
            uniform = torch.full_like(K_final, 1.0 / (H * W))
            K_final = torch.where(degenerate.unsqueeze(1), uniform, K_final)
        K_final = K_final.to(x.dtype)
        # cross-attn
        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)
        kv = self.kv_linear(inst_tokens).view(B, N_inst, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(self.head_dim)  # [B, Hh, N, N_inst]
        if instance_mask is not None:
            pad = (instance_mask <= 0.5).unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad, float("-inf"))
            all_masked = (instance_mask <= 0.5).all(dim=-1)
            if all_masked.any():
                attn[all_masked] = 0.0
        attn = F.softmax(attn, dim=-1)
        
        if attn.shape[2] != K_final.shape[1]:
            print(f"SHAPE MISMATCH DETECTED in GeometryAwareLocalAttention:")
            print(f"  x shape: {x.shape}")
            print(f"  inst_tokens shape: {inst_tokens.shape}")
            print(f"  attn shape: {attn.shape}")
            print(f"  K_final shape: {K_final.shape}")
            print(f"  K_final.unsqueeze(1) shape: {K_final.unsqueeze(1).shape}")
            print(f"  N calculated: {N}")
            print(f"  H, W calculated: {H}, {W}")
        
        
        attn = attn * K_final.unsqueeze(1) # attn modulation
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        x_out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        x_out = x_out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        x_out = self.proj(x_out)
        return x_out

