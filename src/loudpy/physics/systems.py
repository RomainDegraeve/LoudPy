"""
Assembled systems of an axisymmetric model, blocked DOFs eliminated: what the studies solve.

    s = assemble(model.parts)               structure: K, M, C, F (coil force) on the free DOFs
    s = assemble(model.parts, air=True)     + air: H, Q, ρ and the coupling Cam
    s = assemble_fourier(model.parts, m)    structure, azimuthal order m (3 DOF per node)
    dtn = dtn_system(s, parts, ...)         structure + air + exact DtN radiation, symmetric u–ψ–φ form

`s.meca` / `s.acou` are the assemblers: node numbering (unique_tags), expand() from the free DOFs.
No study logic and no files here.
"""
from __future__ import annotations

import logging
from typing import Any
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from loudpy.physics.registry import select
from loudpy.errors import ModelError

log = logging.getLogger("loudpy.physics")


@dataclass
class System:
    """Operators on the free structural DOFs (and the air DOFs when assembled with air)."""
    meca: Any
    K: sp.spmatrix
    M: sp.spmatrix
    C: sp.spmatrix | None
    F: np.ndarray | None = None
    acou: Any = None
    H: sp.spmatrix | None = None
    Q: sp.spmatrix | None = None
    rho: Any = None
    Cam: sp.spmatrix | None = None
    info: dict = field(default_factory=dict)

    @property
    def n_meca(self) -> int:
        """Number of structure DOFs."""
        return self.K.shape[0]

    @property
    def n_acou(self) -> int:
        """Number of air DOFs (0 without air)."""
        return 0 if self.H is None else self.H.shape[0]

    def speed_of_sound(self) -> float:
        """The speed of sound of the air regions (one value required)."""
        cs = {float(sd.c) for sd in self.acou.subdomains if getattr(sd, "c", None) is not None}
        if len(cs) != 1:
            raise ModelError(f"one speed of sound is needed in the air, found {cs}")
        return cs.pop()


def assemble(parts, air: bool = False) -> System:
    """Structural operators (+ air and coupling with air=True), blocked DOFs eliminated."""
    from loudpy.physics.acoustic import AcouAssembler
    from loudpy.physics.coupling import FsiAssembler
    from loudpy.physics.forcing import ForceAssembler
    from loudpy.physics.structural import MecaAssembler
    meca = MecaAssembler(parts)
    K, M, C = meca.assemble()
    F = ForceAssembler(parts, meca).assemble() if select(parts, "coil") else None
    acou = H = Q = rho = Cam = None
    if air:
        acou = AcouAssembler(parts)
        H, Q, rho = acou.assemble()
        Cam = FsiAssembler(parts, meca, acou).assemble()            # (n_acou, n_meca)
    K, M, C, F, Cam = meca.apply_blocked_boundaries(parts, K, M, C, F, Cam)
    return System(meca, K, M, C, F, acou, H, Q, rho, Cam)


def assemble_fourier(parts, m: int) -> System:
    """Structural operators of azimuthal order m ≥ 1 (3 DOF per node), constraints applied."""
    from loudpy.physics.structural import MecaAssemblerFourier
    asm = MecaAssemblerFourier(parts, m)
    K, M, C = asm.apply_blocked_boundaries(parts, *asm.assemble())
    return System(asm, K, M, C)


def assemble_acoustic(parts) -> System:
    """Air operators only (rigid structure): H, Q, ρ."""
    from loudpy.physics.acoustic import AcouAssembler
    acou = AcouAssembler(parts)
    H, Q, rho = acou.assemble()
    return System(None, None, None, None, acou=acou, H=H, Q=Q, rho=rho)


def dtn_system(s: System, parts, center, f_max: float, half: bool = True, n_max: int | None = None):
    """
    Structure + air + exact DtN radiation in the symmetric u–ψ–φ form (p = −sψ, φ = auxiliary
    DtN unknowns):  (s² Ms + s Cs + Ks) [u; ψ; φ] = [F; 0; 0]

      Ms = diag(M, −Q/ρ, 0)
      Cs = [[C, Camᵀ, 0], [Cam, −Σ β_n b_n b_nᵀ/(ρc), 0], [0, 0, −β/(ρc) I]]
      Ks = [[K, 0, 0], [0, −H/ρ − Σ β_n b_n b_nᵀ/(ρR), G], [0, Gᵀ, diag(β z_i/(ρR))]]
      G[:, i] = β_n √z_i b_n/(ρR)  for the i-th root z_i of order n

    (Cam with the outward normal of the air, the same air formulation in every region: see
    physics.coupling.) One DtN operator per radiation arc (e.g. front and rear hemispheres), each
    with its own centre, radius, orders and auxiliary unknowns. center: (0, z0) for every arc,
    {arc name: (0, z0)}, or None
    (each arc: centre on the axis at the height of its node farthest from the axis).
    Returns (Ks, Ms, Cs, info) with info: nm, na, n_aux, arcs, c.
    """
    from loudpy.physics.radiation import dtn_boundary_vectors, dtn_orders, reverse_bessel_roots
    from loudpy.solvers.eigen import to_csc_complex
    R: Any = s.C
    rho = s.rho
    if np.abs(R).sum() < 1e-30:
        R = None
    nm, na = s.n_meca, s.n_acou
    c = s.speed_of_sound()
    arcs = select(parts, "radiation")
    if not arcs:
        raise ModelError("the model has no radiation boundary (lp.Radiation)")

    Dk = sp.csr_matrix((na, na))
    Dc = sp.csr_matrix((na, na))
    G_rows, G_cols, G_vals = [], [], []
    z_aux: list = []
    beta_aux: list = []
    R_aux: list = []
    info = []
    for bd in arcs:
        xy = np.asarray(bd.node_coords).reshape(len(bd.node_tags), -1)[:, :2]
        if isinstance(center, dict) and bd.name in center:
            ctr = tuple(center[bd.name])
        elif center is not None and not isinstance(center, dict):
            ctr = tuple(center)
        else:
            ctr = (0.0, float(xy[np.argmax(xy[:, 0]), 1]))
        _, R_b = dtn_boundary_vectors(bd.edges, bd.node_tags, bd.node_coords, s.acou.tag_to_local, na, ctr, [0])
        orders, _ = dtn_orders(R_b, f_max, c, half, n_max)
        Bvec, R_b = dtn_boundary_vectors(bd.edges, bd.node_tags, bd.node_coords, s.acou.tag_to_local, na, ctr,
                                         orders)
        bnd = np.flatnonzero(np.any(Bvec != 0, axis=0))
        dk = np.zeros((bnd.size, bnd.size))
        dc = np.zeros_like(dk)
        n_aux0 = len(z_aux)
        for j, n in enumerate(orders):
            beta = (2 * n + 1) / ((2.0 if half else 4.0) * np.pi * R_b ** 2)
            b = Bvec[j, bnd]
            dk -= beta / (rho * R_b) * np.outer(b, b)
            dc -= beta / (rho * c) * np.outer(b, b)
            for zi in reverse_bessel_roots(n):
                G_rows.append(bnd)
                G_cols.append(np.full(bnd.size, len(z_aux)))
                G_vals.append(beta * np.sqrt(zi) / (rho * R_b) * b)
                z_aux.append(zi)
                beta_aux.append(beta)
                R_aux.append(R_b)
        ii, jj = np.meshgrid(bnd, bnd, indexing="ij")
        Dk = Dk + sp.csr_matrix((dk.ravel(), (ii.ravel(), jj.ravel())), shape=(na, na))
        Dc = Dc + sp.csr_matrix((dc.ravel(), (ii.ravel(), jj.ravel())), shape=(na, na))
        info.append(dict(name=bd.name, R=R_b, center=ctr, orders=orders,
                         n_aux=len(z_aux) - n_aux0, n_bnd=bnd.size))
        log.info("DtN arc %s: R = %.4f m, centre %s, orders %d..%d (%s), "
                 "%d auxiliary unknowns, %d boundary nodes", bd.name, R_b, ctr, orders[0], orders[-1],
                 "even, baffled hemisphere" if half else "all, sphere", len(z_aux) - n_aux0, bnd.size)

    n_aux = len(z_aux)
    z_arr, beta_arr, R_arr = np.array(z_aux), np.array(beta_aux), np.array(R_aux)
    G = (sp.csr_matrix((np.concatenate(G_vals), (np.concatenate(G_rows), np.concatenate(G_cols))),
                       shape=(na, n_aux)) if n_aux else sp.csr_matrix((na, 0)))
    pieces = dict(Dk=Dk, Dc=Dc, G=G,
                  Zaa=sp.diags(beta_arr * z_arr / (rho * R_arr)) if n_aux else None,
                  Caa=sp.diags(-beta_arr / (rho * c)) if n_aux else None, n_aux=n_aux)
    from loudpy.physics.builder import symmetric_builder
    s_damped = s if R is not None else System(s.meca, s.K, s.M, None, s.F, s.acou, s.H, s.Q, s.rho, s.Cam)
    builder = symmetric_builder(s_damped, parts, pieces)
    K0, C0, M0 = builder.polynomial()                          # A(s) = K0 + s C0 + s² M0
    Ks, Ms, Cs = to_csc_complex(K0, M0, C0)
    log.info("structure + air + DtN: %d structural, %d air DOFs, %d arc(s), %d auxiliary unknowns",
             nm, na, len(arcs), n_aux)
    return Ks, Ms, Cs, dict(nm=nm, na=na, n_aux=n_aux, arcs=info, c=c, builder=builder)


def nodal(raw, n_nodes: int, dpn: int) -> np.ndarray:
    """Full DOF vectors (..., n_nodes·dpn) → node-indexed (..., n_nodes[, dpn]) (assembler node order)."""
    raw = np.asarray(raw)
    if dpn == 1:
        return raw[..., :n_nodes]
    return raw[..., :dpn * n_nodes].reshape(raw.shape[:-1] + (n_nodes, dpn))


def physics_mesh(parts, roles, node_tags):
    """Mesh of the regions with `roles`, nodes in the order `node_tags`, one group per region."""
    from loudpy.geometry.mesh import Mesh, MeshGroup
    regions = select(parts, *roles)
    coords = {}
    for p in regions:
        for t, c in zip(np.asarray(p.node_tags), np.asarray(p.node_coords)):
            coords[int(t)] = c[:2]
    tags = np.asarray(node_tags)
    groups = {p.name: MeshGroup(p.name, 2, "tri6", np.asarray(p.tri)) for p in regions}
    return Mesh(tags, np.array([coords[int(t)] for t in tags]), groups)


def meshes(s: System, parts) -> dict:
    """{"solid": mesh of the structural DOFs, "air": mesh of the air DOFs} of an assembled system."""
    out = {}
    if s.meca is not None:
        out["solid"] = physics_mesh(parts, ("solid",), s.meca.unique_tags)
    if s.acou is not None:
        out["air"] = physics_mesh(parts, ("air", "pml"), s.acou.unique_tags)
    return out
