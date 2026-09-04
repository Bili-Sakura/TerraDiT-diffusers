"""Visualization of TerraDiT conditioning inputs.

Renders the structured conditioning (sigma point prompts, omega instance geometry)
over the generated 256x256 tile so you can see exactly which points / primitives
and OSM tags were fed to the model. All coordinates are in tile pixel space
[0, 256]; sigma point coords are (row, col), omega geometry is (x, y).

Omega primitives follow a "derive-down" priority per instance: a polygon (or
polyline) fully implies its bounding box and centroid point, so only the richest
*active* primitive is drawn -- polygon > polyline > bbox > point. Which primitives
are "active" is decided by the demo's --condition-type (the GALA dropout mask).
"""
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

RES = 256

# per-primitive colors
C_POLYGON = (255, 80, 80)
C_POLYLINE = (80, 200, 255)
C_BBOX = (255, 215, 0)
C_POINT = (80, 255, 120)


def tensor_to_pil(img):
    """img: [3, H, W] float in [0, 1] -> PIL RGB image."""
    arr = (img.detach().clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def save_images(imgs, paths):
    """Save a batch of ``[3,H,W]`` float tensors in ``[0, 1]`` (or PIL images)."""
    for img, path in zip(imgs, paths):
        if torch.is_tensor(img):
            torchvision.utils.save_image(img, path)
        else:
            img.save(path)


def _font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _label(draw, xy, text, color):
    if not text:
        return
    x, y = xy
    draw.text((x + 4, y - 6), text, fill=color, font=_font())


def active_from_dropouts(dropout_probs):
    """Map a GALA dropout_probs vector -> which modalities survive (are 'provided').

    dropout_probs is [p_polygon, p_polyline, p_bbox, p_point] drop-probabilities, or
    None for "keep everything". A modality is active iff its drop-prob < 1.0.
    """
    keys = ("polygon", "polyline", "bbox", "point")
    if dropout_probs is None:
        return {k: True for k in keys}
    return {k: float(dropout_probs[i]) < 1.0 for i, k in enumerate(keys)}


def render_sigma_inputs(base_img, coords, mask, tags, out_path):
    """Overlay sigma point prompts on the generated tile.

    base_img: [3, 256, 256] in [0, 1]. coords: [P, 2] as (row, col).
    mask: [P] with 0 = real point, 1 = padding. tags: list[str] of length P.
    Returns the number of points drawn.
    """
    im = tensor_to_pil(base_img).copy()
    d = ImageDraw.Draw(im)
    r = 3
    n = 0
    for i in range(coords.shape[0]):
        if int(mask[i]) != 0:  # padding slot
            continue
        row, col = float(coords[i, 0]), float(coords[i, 1])
        x, y = col, row  # (row, col) -> (x, y)
        d.ellipse([x - r, y - r, x + r, y + r], fill=C_POINT, outline=(0, 0, 0))
        _label(d, (x, y), tags[i] if i < len(tags) else "", C_POINT)
        n += 1
    im.save(out_path)
    return n


def render_omega_inputs(base_img, *, polygon_xy, polygon_xy_mask, polyline_xy,
                        polyline_xy_mask, bbox_xyxy, point_xy, instance_mask,
                        tags, active, out_path):
    """Overlay omega instance geometry on the generated tile (derive-down priority).

    Per active instance, draw exactly one primitive: polygon if a polygon is active
    and present, else polyline, else bbox, else point. `active` is the dict from
    active_from_dropouts(). All geometry tensors are (x, y) in [0, 256].
    Returns a dict of per-primitive draw counts.
    """
    im = tensor_to_pil(base_img).copy()
    d = ImageDraw.Draw(im)
    counts = dict(polygon=0, polyline=0, bbox=0, point=0)
    r = 3
    for j in range(instance_mask.shape[0]):
        if float(instance_mask[j]) < 0.5:
            continue
        tag = tags[j] if j < len(tags) else ""
        has_pg = bool(polygon_xy_mask[j].any())
        has_pl = bool(polyline_xy_mask[j].any())

        if active["polygon"] and has_pg:
            pts = polygon_xy[j][polygon_xy_mask[j]]
            xy = [(float(p[0]), float(p[1])) for p in pts]
            if len(xy) >= 2:
                d.line(xy + [xy[0]], fill=C_POLYGON, width=2)  # closed ring
                _label(d, xy[0], tag, C_POLYGON)
                counts["polygon"] += 1
        elif active["polyline"] and has_pl:
            pts = polyline_xy[j][polyline_xy_mask[j]]
            xy = [(float(p[0]), float(p[1])) for p in pts]
            if len(xy) >= 2:
                d.line(xy, fill=C_POLYLINE, width=2)  # open path
                _label(d, xy[0], tag, C_POLYLINE)
                counts["polyline"] += 1
        elif active["bbox"]:
            x0, y0, x1, y1 = [float(v) for v in bbox_xyxy[j]]
            d.rectangle([x0, y0, x1, y1], outline=C_BBOX, width=2)
            _label(d, (x0, y0), tag, C_BBOX)
            counts["bbox"] += 1
        elif active["point"]:
            x, y = float(point_xy[j][0]), float(point_xy[j][1])
            d.ellipse([x - r, y - r, x + r, y + r], fill=C_POINT, outline=(0, 0, 0))
            _label(d, (x, y), tag, C_POINT)
            counts["point"] += 1
    im.save(out_path)
    return counts
