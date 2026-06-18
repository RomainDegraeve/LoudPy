import numpy as np

def _local_coords(node_tags, node_coords, flat_tags, n_e, n_pts, dim=None):
    """Map global tags → local coordinates, reshape to (n_e, n_pts, dim)."""
    tags = np.asarray(node_tags)
    coords = np.asarray(node_coords)
    sort_idx = np.argsort(tags)
    local_idx = sort_idx[np.searchsorted(tags[sort_idx], flat_tags)]
    ec = coords[local_idx].reshape(n_e, n_pts, -1)
    if dim is not None:
        ec = ec[:, :, :dim]
    return np.ascontiguousarray(ec)