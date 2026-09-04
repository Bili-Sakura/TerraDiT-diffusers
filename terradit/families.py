"""Per-family model conditioning flags, shared by training and inference."""

# SiT construction flags that distinguish the three released model families.
# legacy=False -> adaLN conditioning t = y_pooled + t_embed  (intended; all families)
# legacy=True  -> adaLN conditioning t = t_embed             (earlier training runs; the
#                 released weights all use legacy=False). Loadable via the --legacy override.
FAMILY_CONSTRUCT = {
    "alpha": dict(condition_type="text", geolocation=False, point_prompts=False, omega=False, legacy=False),
    "sigma": dict(condition_type="text", geolocation=True, point_prompts=True, omega=False, legacy=False),
    "omega": dict(condition_type="text", geolocation=True, point_prompts=False, omega=True, legacy=False),
}
