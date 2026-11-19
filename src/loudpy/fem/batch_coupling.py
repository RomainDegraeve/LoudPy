from numba import njit, prange
import numpy as np



@njit(parallel=False, cache=True, fastmath=True, boundscheck=False)
def _compute_signs_jit(edges, ec, indptr, neigh_tris, neigh_centers):
    n_e = edges.shape[0]
    signs = np.ones(n_e, dtype=np.float64)
    
    print("\n" + "="*50)
    print("FSI NORMAL ORIENTATION DEBUG")
    print("="*50)

    for e in range(n_e):
        t0 = edges[e, 0]
        t1 = edges[e, 1]

        # 1. Edge midpoint (Local center)
        lc_r = (ec[e, 0, 0] + ec[e, 1, 0] + ec[e, 2, 0]) / 3.0
        lc_z = (ec[e, 0, 1] + ec[e, 1, 1] + ec[e, 2, 1]) / 3.0

        # 2. Mathematical Normal (dz, -dr)
        dr = ec[e, 1, 0] - ec[e, 0, 0]
        dz = ec[e, 1, 1] - ec[e, 0, 1]
        nr = dz
        nz = -dr

        # 3. Find Fluid Neighbor Centroid
        found = False
        cr, cz = 0.0, 0.0
        for k in range(indptr[t0], indptr[t0 + 1]):
            hit = False
            for j in range(neigh_tris.shape[1]):
                if neigh_tris[k, j] == t1:
                    hit = True
                    break
            if hit:
                cr = neigh_centers[k, 0]
                cz = neigh_centers[k, 1]
                found = True
                break
        
        if not found:
            print("Edge", e, ": No fluid neighbor found!")
            continue

        # 4. Vector from Edge to Fluid Centroid
        vec_to_fluid_r = cr - lc_r
        vec_to_fluid_z = cz - lc_z

        # 5. Orientation check (Dot product)
        # Positive means the normal (dz, -dr) points TOWARDS the fluid
        dot = (nr * vec_to_fluid_r) + (nz * vec_to_fluid_z)
        
        if dot > 0.0:
            signs[e] = -1.0
        else:
            signs[e] = 1.0

        # --- CLEAN DEBUG PRINT ---
        print("Edge:", e, "Nodes:", t0, "->", t1)
        print("  Midpoint:  (", lc_r, ",", lc_z, ")")
        print("  Fluid Ctr: (", cr, ",", cz, ")")
        print("  Raw Normal:(", nr, ",", nz, ")")
        print("  Dot Product:", dot, " -> Sign:", signs[e])
        print("-" * 30)

    return signs






@njit(parallel=False, cache=True, fastmath=True, boundscheck=False)
def _compute_signs_jit(edges, ec, indptr, neigh_tris, neigh_centers):
    n_e = edges.shape[0]
    signs = np.ones(n_e, dtype=np.float64)

    for e in range(n_e):
        t0 = edges[e, 0]
        t1 = edges[e, 1]

        # Edge midpoint (use the two endpoints, NOT the 3-node average,
        # because node 3 is the mid-edge node and skews nothing here,
        # but endpoints are cleaner)
        lc_r = 0.5 * (ec[e, 0, 0] + ec[e, 1, 0])
        lc_z = 0.5 * (ec[e, 0, 1] + ec[e, 1, 1])

        dr = ec[e, 1, 0] - ec[e, 0, 0]
        dz = ec[e, 1, 1] - ec[e, 0, 1]
        nr = dz
        nz = -dr

        found = False
        cr, cz = 0.0, 0.0
        for k in range(indptr[t0], indptr[t0 + 1]):
            hit = False
            for j in range(neigh_tris.shape[1]):
                if neigh_tris[k, j] == t1:
                    hit = True
                    break
            if hit:
                cr = neigh_centers[k, 0]
                cz = neigh_centers[k, 1]
                found = True
                break   # ← CRITICAL FIX

        if not found:
            continue

        dot = nr * (cr - lc_r) + nz * (cz - lc_z)
        if dot > 0.0:
            signs[e] = 1.0
        else:
            signs[e] = -1.0

    return signs




@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _fsi_local_to_V(out_x, out_y, signs, V):
    """
    Vectorized interleave + sign application, written directly into V.
    out_x, out_y : (n_e, n_p*n_p) flat
    V            : (n_e * n_p * (n_p*2),) flat
    """
    n_e = out_x.shape[0]
    npp = out_x.shape[1]                       # n_p * n_p
    n_p = 0
    # infer n_p from npp (small, computed once)
    while n_p * n_p != npp:
        n_p += 1

    block = n_p * (n_p * 2)
    for e in prange(n_e):
        s = signs[e]
        base = e * block
        for i in range(n_p):
            row_off = base + i * (n_p * 2)
            xrow = i * n_p
            for j in range(n_p):
                V[row_off + 2 * j]     = s * out_x[e, xrow + j]
                V[row_off + 2 * j + 1] = s * out_y[e, xrow + j]


