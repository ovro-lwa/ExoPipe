"""Alignment flows for `exopipe3 flow` (steps/flow.py): batch_flow_align, the per-night co-add flow batch (B2), run as
a SUBPROCESS under py310_orca (image_plane_correction + JAX, GPU) by masters' co-add and the search identification
module. Never import a flow engine at module level from a py38 process."""
