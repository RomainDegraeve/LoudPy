"""
Derived values of the 2D axisymmetric frequency results (harmonic amplitudes, time factor e^{jωt}).

    volume_velocity(res, on)          Q = ∫ jω u·n dA over a boundary [m³/s], n out of the structure
    radiated_power(res, on=…)         P = ½ Re ∫ p* jω u·n dA through a boundary of the structure [W]
    radiated_power(res, radius=…)     P = ∮ ½ Re(p v*)·n dA through a sphere in the air (v = −∇p / (jωρ))
    thiele_small(res, on=…)           small-signal parameters from the impedance (voltage drive) and the
                                      volume velocity: Re, Le, n, Bl (drive); fs, Mms, Cms, Rms, Qms, Qes, Qts,
                                      Sd, Vas

Boundary integrals use the curved edges of the mesh (their Lagrange basis, 4-point Gauss) with dA = 2πr ds;
the normal points out of the solid region next to each edge (into the air for a fluid–structure boundary).
"""
from __future__ import annotations

import numpy as np

from loudpy.errors import ResultsError

_GAUSS = np.polynomial.legendre.leggauss(4)


def _edges(res, on):
    if on not in res.boundaries:
        raise ResultsError(f"no boundary {on!r} in these results (boundaries: {list(res.boundaries)})")
    e = np.asarray(res.boundaries[on].get("edges", []))
    if not len(e):
        raise ResultsError(f"the boundary {on!r} has no edges stored")
    return e


def edge_quadrature(mesh, edges):
    """
    Gauss points of the edges (rows of node tags, gmsh line order) on `mesh`: basis values N (n_e, n_g, k),
    rows (n_e, k) of the nodes, points (n_e, n_g, 2), unit normals out of the adjacent solid cell (n_e, n_g, 2)
    and weights 2πr ds (n_e, n_g).
    """
    from loudpy.fem.reference import element_for_cells
    k = edges.shape[1]
    elem = element_for_cells(1, k)
    xg, wg = _GAUSS
    N, dN = elem.tabulate(xg[:, None])                         # (n_g, k), (n_g, k, 1)
    rows = mesh._rows(edges)
    X = mesh.coords[rows]                                      # (n_e, k, 2)
    pts = np.einsum("gk,ekd->egd", N, X)
    t = np.einsum("gk,ekd->egd", dN[..., 0], X)
    ds = np.linalg.norm(t, axis=2)
    n = np.stack([t[..., 1], -t[..., 0]], axis=-1) / ds[..., None]
    third = _third_corners(mesh, edges)                        # a point of the adjacent cell, per edge
    mid = X[:, :2].mean(axis=1)
    sign = np.sign(np.einsum("ed,ed->e", n[:, 0], mid - third))
    sign[sign == 0] = 1.0
    n = n * sign[:, None, None]
    return N, rows, pts, n, ds * wg[None, :] * 2 * np.pi * pts[..., 0]


def _third_corners(mesh, edges):
    """For each edge (corners first), the centre of the (single) 2D cell of `mesh` having both corners."""
    want = {(min(a, b), max(a, b)): i for i, (a, b) in enumerate(np.asarray(edges)[:, :2].tolist())}
    out = np.full((len(edges), 2), np.nan)
    for g in mesh.groups.values():
        if g.dim != 2:
            continue
        cells = np.asarray(g.cells)
        nc = 3 if cells.shape[1] in (3, 6, 10) else 4
        corners = cells[:, :nc]
        for j in range(nc):
            a, b = corners[:, j], corners[:, (j + 1) % nc]
            for c, (x, y) in enumerate(zip(a.tolist(), b.tolist())):
                i = want.get((min(x, y), max(x, y)))
                if i is not None and np.isnan(out[i, 0]):
                    out[i] = mesh.coords[mesh._rows(corners[c])].mean(axis=0)
    if np.isnan(out[:, 0]).any():
        raise ResultsError("some boundary edges have no adjacent cell in the structure's mesh")
    return out


def _normal_velocity(res, on):
    """(blocks …) per block: Gauss data and u·n at the Gauss points (n_f, n_e, n_g)."""
    edges = _edges(res, on)
    out = []
    for b in res.blocks:
        if "u" not in b.fields:
            raise ResultsError(f"no displacement field in block {b.label!r}")
        mesh = b.mesh_of("u")
        N, rows, pts, n, w = edge_quadrature(mesh, edges)
        U = np.asarray(b.fields["u"])[:, rows, :2]                       # (n_f, n_e, k, 2)
        un = np.einsum("gk,fekd,egd->feg", N, U, n)
        out.append((b, N, w, un, edges))
    return out


def volume_velocity(res, on: str):
    """Q(f) = ∫ jω u·n dA over the boundary `on` [m³/s] (n out of the structure)."""
    from loudpy.results.data import Data
    vals = []
    for b, _, w, un, _ in _normal_velocity(res, on):
        jw = 2j * np.pi * np.asarray(b.values)
        vals.append(jw * np.einsum("feg,eg->f", un, w))
    return Data(np.concatenate(vals), {"f": res.frequencies}, "Q", meta=res._meta(on=on))


def radiated_power(res, on: str | None = None, radius: float | None = None, center=(0.0, 0.0),
                   baffle: float | None = None):
    """Time-averaged acoustic power [W] through the boundary `on` of the structure (from u and p), or through
    a sphere (radius, centre on the axis, optional rigid baffle) in the air (from p and ∇p)."""
    from loudpy.results.data import Data
    if (on is None) == (radius is None):
        raise ResultsError("give either on= (a boundary) or radius= (a sphere in the air)")
    vals = []
    if on is not None:
        for b, N, w, un, edges in _normal_velocity(res, on):
            if "p" not in b.fields:
                raise ResultsError("the power through a boundary needs the pressure (a coupled result)")
            air = b.mesh_of("p")
            P = np.asarray(b.fields["p"])[:, air._rows(edges)]                # (n_f, n_e, k)
            pg = np.einsum("gk,fek->feg", N, P)
            jw = 2j * np.pi * np.asarray(b.values)
            vals.append(0.5 * np.real(np.einsum("feg,feg,eg->f", np.conj(pg), jw[:, None, None] * un, w)))
    else:
        from loudpy.results.farfield import sphere_arc
        assert radius is not None                      # on= or radius= (checked above)
        rho = _air_density(res)
        for b in res.blocks:
            mesh = b.mesh_of("p")
            pts, nrm, ds = sphere_arc(radius, center, baffle, 128)
            W, (Gr, Gz) = mesh.interpolation(pts, gradient=True)
            if ((Gr.getnnz(axis=1) == 0) & (Gz.getnnz(axis=1) == 0)).any():
                raise ResultsError("the sphere leaves the mesh: choose a radius inside the air")
            P = np.asarray(b.fields["p"])
            p = (W @ P.T).T
            dpdn = (Gr @ P.T).T * nrm[:, 0] + (Gz @ P.T).T * nrm[:, 1]
            w = 2 * np.pi * np.asarray(b.values)
            vn = -dpdn / (1j * w[:, None] * rho)
            vals.append(0.5 * np.real(p * np.conj(vn)) @ (ds * 2 * np.pi * pts[:, 0]))
    return Data(np.concatenate(vals), {"f": res.frequencies}, "power",
                meta=res._meta(on=on if on is not None else f"sphere R = {radius:g} m"))


def _air_density(res) -> float:
    groups = res.model.groups if res.model is not None else {}
    rhos = {g["rho"] for g in groups.values() if g.get("role") == "air" and "rho" in g}
    if len(rhos) != 1:
        raise ResultsError("the power through a sphere needs one air density (results of format 3)")
    return rhos.pop()


def thiele_small(res, on: str | None = None, coil: str | None = None, band: float = 1.5) -> dict:
    """
    Small-signal (Thiele–Small) parameters of a voltage-driven result. The motional impedance
    Z_mot = Z − (Re + (jω)ⁿ Le) is fitted as Bl² / (Rms + jωMms + 1/(jωCms)): Mms and Cms by linear least
    squares of Im(Bl²/Z_mot) over fs/band … fs·band around its peak, Rms = Re(Bl²/Z_mot) at fs (the
    mechanical resistance of a FE model varies with frequency: radiation, Rayleigh damping); then fs, Qms,
    Qes, Qts. With `on` (the radiating boundary)
    and the coil boundary: Sd = |Q| / |v_coil| at fs and Vas = ρc² Sd² Cms. Returns a dict (SI units).
    """
    drive = (res.study or {}).get("drive") or {}
    if drive.get("type") != "Voltage":
        raise ResultsError("Thiele–Small parameters need a voltage-driven frequency response (lp.Voltage)")
    Re, Le, n, Bl = (float(drive[k]) for k in ("Re", "Le", "n", "Bl"))
    f = res.frequencies
    w = 2 * np.pi * f
    Z = res.impedance().values
    Zm = Z - (Re + (1j * w) ** n * Le)
    i0 = int(np.argmax(np.abs(Zm)))
    sel = (f >= f[i0] / band) & (f <= f[i0] * band)
    if sel.sum() < 3:
        raise ResultsError("too few frequencies around the resonance to fit the Thiele–Small parameters")
    Y = Bl ** 2 / Zm[sel]                                            # = Rms + j(ω Mms − 1/(ω Cms))
    ws = w[sel]
    A = np.column_stack([ws, -1.0 / ws])
    (Mms, invC), *_ = np.linalg.lstsq(A, Y.imag, rcond=None)
    Cms = 1.0 / invC
    ws0 = float(1.0 / np.sqrt(Mms * Cms))
    Rms = float(np.interp(ws0, ws, Y.real))
    out = dict(Re=Re, Le=Le, n=n, Bl=Bl, fs=ws0 / (2 * np.pi), Mms=float(Mms), Cms=float(Cms), Rms=Rms,
               Qms=float(ws0 * Mms / Rms), Qes=float(ws0 * Mms * Re / Bl ** 2))
    out["Qts"] = out["Qms"] * out["Qes"] / (out["Qms"] + out["Qes"])
    if on is not None:
        coil = coil or _coil_label(res)
        Q = volume_velocity(res, on).values
        k = int(np.argmin(np.abs(f - out["fs"])))
        vc = _coil_velocity(res, coil)[k]
        out["Sd"] = float(np.abs(Q[k]) / np.abs(vc))
        groups = res.model.groups if res.model is not None else {}
        air = [g for g in groups.values() if g.get("role") == "air"]
        if air:
            rho, c = air[0]["rho"], air[0]["c"]
            out["Vas"] = float(rho * c ** 2 * out["Sd"] ** 2 * Cms)
    return out


def _coil_label(res) -> str:
    items = (res.model["boundaries"] if res.model is not None else [])
    coils = [b["name"] for b in items if b.get("type") == "Coil"]
    if len(coils) != 1:
        raise ResultsError("give the coil boundary: coil=")
    return coils[0]


def _coil_velocity(res, coil) -> np.ndarray:
    """Mean axial velocity jω u_z of the coil boundary's nodes over frequency."""
    tags = np.asarray(res.boundaries[coil]["node_tags"])
    out = []
    for b in res.blocks:
        mesh = b.mesh_of("u")
        uz = np.asarray(b.fields["u"])[:, mesh._rows(tags), 1].mean(axis=1)
        out.append(2j * np.pi * np.asarray(b.values) * uz)
    return np.concatenate(out)
