import torch
import numpy as np


def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    return 2 * t_cur


def euler_sampler(
        model,
        latents,
        y,
        y_pooled=None,
        loc_embed=None,
        y_null=None,
        y_null_pooled=None,
        point_prompts=None,
        pos=None,
        mask=None,
        inst_text_embed=None,
        polygon_xy=None,
        polygon_xy_mask=None,
        polyline_xy=None,
        polyline_xy_mask=None,
        bbox_xyxy=None,
        point_xy=None,
        format_mask=None,
        instance_mask=None,
        dropout_probs=None,
        num_steps=50,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear", # not used, just for compatability
        ):
    # setup conditioning
    # if cfg_scale > 1.0:
        # y_null = torch.tensor([1000] * y.size(0), device=y.device)
    _dtype = latents.dtype    
    t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
                y_pool = torch.cat([y_pooled, y_null_pooled], dim=0)
                if inst_text_embed is not None:
                    inst_text_embed_cfg = torch.cat([inst_text_embed] * 2, dim=0)
                    polygon_xy_cfg = torch.cat([polygon_xy] * 2, dim=0)
                    polygon_xy_mask_cfg = torch.cat([polygon_xy_mask] * 2, dim=0)
                    polyline_xy_cfg = torch.cat([polyline_xy] * 2, dim=0)
                    polyline_xy_mask_cfg = torch.cat([polyline_xy_mask] * 2, dim=0)
                    bbox_xyxy_cfg = torch.cat([bbox_xyxy] * 2, dim=0)
                    point_xy_cfg = torch.cat([point_xy] * 2, dim=0)
                    format_mask_cfg = torch.cat([format_mask] * 2, dim=0)
                    instance_mask_cfg = torch.cat([instance_mask] * 2, dim=0)
                else:
                    inst_text_embed_cfg = polygon_xy_cfg = polygon_xy_mask_cfg = None
                    polyline_xy_cfg = polyline_xy_mask_cfg = bbox_xyxy_cfg = None
                    point_xy_cfg = format_mask_cfg = instance_mask_cfg = None
            else:
                model_input = x_cur
                y_cur = y
                y_pool = y_pooled
                inst_text_embed_cfg = inst_text_embed
                polygon_xy_cfg = polygon_xy
                polygon_xy_mask_cfg = polygon_xy_mask
                polyline_xy_cfg = polyline_xy
                polyline_xy_mask_cfg = polyline_xy_mask
                bbox_xyxy_cfg = bbox_xyxy
                point_xy_cfg = point_xy
                format_mask_cfg = format_mask
                instance_mask_cfg = instance_mask
                
            kwargs = dict(
                y=y_cur,
                y_pooled=y_pool,
                loc_embed=loc_embed,
                point_prompts=point_prompts,
                pos=pos,
                mask=mask,
                inst_text_embed=inst_text_embed_cfg,
                polygon_xy=polygon_xy_cfg,
                polygon_xy_mask=polygon_xy_mask_cfg,
                polyline_xy=polyline_xy_cfg,
                polyline_xy_mask=polyline_xy_mask_cfg,
                bbox_xyxy=bbox_xyxy_cfg,
                point_xy=point_xy_cfg,
                format_mask=format_mask_cfg,
                instance_mask=instance_mask_cfg,
                dropout_probs=dropout_probs,
            )
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            d_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                )[0].to(torch.float64)
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)                
            x_next = x_cur + (t_next - t_cur) * d_cur
            if heun and (i < num_steps - 1):
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    model_input = torch.cat([x_next] * 2)
                    y_cur = torch.cat([y, y_null], dim=0)
                else:
                    model_input = x_next
                    y_cur = y
                kwargs = dict(
                    y=y_cur,
                    y_pooled=y_pool,
                    loc_embed=loc_embed,
                    point_prompts=point_prompts,
                    pos=pos,
                    mask=mask,
                    inst_text_embed=inst_text_embed_cfg,
                    polygon_xy=polygon_xy_cfg,
                    polygon_xy_mask=polygon_xy_mask_cfg,
                    polyline_xy=polyline_xy_cfg,
                    polyline_xy_mask=polyline_xy_mask_cfg,
                    bbox_xyxy=bbox_xyxy_cfg,
                    point_xy=point_xy_cfg,
                    format_mask=format_mask_cfg,
                    instance_mask=instance_mask_cfg,
                    dropout_probs=dropout_probs_,
                )
                time_input = torch.ones(model_input.size(0)).to(
                    device=model_input.device, dtype=torch.float64
                    ) * t_next
                d_prime = model(
                    model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                    )[0].to(torch.float64)
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    d_prime_cond, d_prime_uncond = d_prime.chunk(2)
                    d_prime = d_prime_uncond + cfg_scale * (d_prime_cond - d_prime_uncond)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
                
    return x_next


def euler_inpaint_sampler(
        model,
        latents,
        y,
        y_pooled=None,
        loc_embed=None,
        y_null=None,
        y_null_pooled=None,
        point_prompts=None,
        pos=None,
        inst_text_embed=None,
        polygon_xy=None,
        polygon_xy_mask=None,
        polyline_xy=None,
        polyline_xy_mask=None,
        bbox_xyxy=None,
        point_xy=None,
        format_mask=None,
        instance_mask=None,
        dropout_probs=None,
        num_steps=50,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        inpaint=False,
        ref_latents=None,
        mask_latents=None,
        **kwargs
        ):
    _dtype = latents.dtype    
    t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device

    # if inpaint:
    #     # If latents (xT) was passed in as random noise, we can use it as the noise basis
    #     ref_noise = latents.clone().to(torch.float64)
    
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
                y_pool = torch.cat([y_pooled, y_null_pooled], dim=0)
                
                def cfg_cat(x): return torch.cat([x] * 2, dim=0) if x is not None else None
                
                inst_text_embed_cfg = cfg_cat(inst_text_embed)
                polygon_xy_cfg = cfg_cat(polygon_xy)
                polygon_xy_mask_cfg = cfg_cat(polygon_xy_mask)
                polyline_xy_cfg = cfg_cat(polyline_xy)
                polyline_xy_mask_cfg = cfg_cat(polyline_xy_mask)
                bbox_xyxy_cfg = cfg_cat(bbox_xyxy)
                point_xy_cfg = cfg_cat(point_xy)
                format_mask_cfg = cfg_cat(format_mask)
                instance_mask_cfg = cfg_cat(instance_mask)
                loc_embed_cfg = cfg_cat(loc_embed)
            else:
                model_input = x_cur
                y_cur = y
                y_pool = y_pooled
                inst_text_embed_cfg = inst_text_embed
                polygon_xy_cfg = polygon_xy
                polygon_xy_mask_cfg = polygon_xy_mask
                polyline_xy_cfg = polyline_xy
                polyline_xy_mask_cfg = polyline_xy_mask
                bbox_xyxy_cfg = bbox_xyxy
                point_xy_cfg = point_xy
                format_mask_cfg = format_mask
                instance_mask_cfg = instance_mask
                loc_embed_cfg = loc_embed

            model_kwargs = dict(
                y=y_cur, y_pooled=y_pool, loc_embed=loc_embed_cfg, point_prompts=point_prompts,
                pos=pos, inst_text_embed=inst_text_embed_cfg, polygon_xy=polygon_xy_cfg,
                polygon_xy_mask=polygon_xy_mask_cfg, polyline_xy=polyline_xy_cfg,
                polyline_xy_mask=polyline_xy_mask_cfg, bbox_xyxy=bbox_xyxy_cfg,
                point_xy=point_xy_cfg, format_mask=format_mask_cfg,
                instance_mask=instance_mask_cfg, dropout_probs=dropout_probs,
            )

            time_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_cur
            
            # Predict
            d_cur = model(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **model_kwargs)[0].to(torch.float64)

            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)                
            
            # Euler Step
            x_next = x_cur + (t_next - t_cur) * d_cur

            if inpaint:
                noise = torch.randn_like(ref_latents) 
                ref_noised = (1 - t_next) * ref_latents + t_next * noise
                x_next = (1 - mask_latents) * ref_noised + mask_latents * x_next
            # if inpaint:
            #     ref_noised = (1 - t_next) * ref_latents + t_next * ref_noise
            #     x_next = (1 - mask_latents) * ref_noised + mask_latents * x_next
                
    return x_next

def euler_maruyama_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        ):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
            
    _dtype = latents.dtype
    
    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y            
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            diffusion = compute_diffusion(t_cur)            
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            # compute drift
            v_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                )[0].to(torch.float64)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

            x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps
    
    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y            
    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
        ) * t_cur
    
    # compute drift
    v_cur = model(
        model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
        )[0].to(torch.float64)
    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur
                    
    return mean_x
