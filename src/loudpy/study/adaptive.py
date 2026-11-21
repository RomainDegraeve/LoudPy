"""
Mesh adaptation and convergence studies, on copies of the model (the user's model is never changed).

    ad = lp.Adaptive(lp.Modes(model), target=0.02, max_iterations=4)
    res = ad.solve(f_max=3000)            # the study's results on the last mesh
    print(ad)                             # one row per mesh: DOFs, cells, estimate (solid, air)
    ad.history                            # the same rows as dicts (also in res.meta["adaptive"], JSON)

    table = lp.convergence(lp.Modes(model), factors=(1, 0.7, 0.5), f_max=3000)
    print(table)                          # the quantity vs DOFs, observed orders, Richardson extrapolation
    table.values, table.order, table.extrapolated, table.errors, table.expected

Adaptive: each mesh is solved, the error is estimated per cell from the results (loudpy.fem.estimate):
  - solids: the energy norm of the strain recovery, ∫ (ε* − ε_h)ᴴ D (ε* − ε_h) 2πr, relative to ∫ ε_hᴴ D ε_h 2πr
  - air (outside the PML): the pressure-gradient recovery, ∫ |G*p − ∇p|² 2πr, relative to ∫ |∇p|² 2πr
for every entry (frequency, mode); a cell keeps the largest relative indicator over the entries. The estimate
of a mesh is the largest relative estimate √(Σ η² / ‖·‖²) over the entries and the two physics. While it is
above `target`, new sizes for 0.5 × target are computed by equidistribution (loudpy.fem.adapt, the factor per
step within [0.3, 2]) and the model is remeshed with
them as a gmsh size view (model.mesh(size_field=…)): <mesh stem>_adapt<i>.msh / .pos next to the model's mesh.
The regions' mesh sizes stay the coarsest sizes (the view refines where it is smaller). PML cells are not
estimated (their mesh is the PML's own). Axisymmetric fields (m = 0) only.

convergence: the model is meshed with every region and mesh-control size multiplied by each factor (factors
decreasing: finer meshes), the study solved on each mesh and `quantity(results)` (default for lp.Modes: the
mode frequencies) evaluated. With h = N^(-1/2) (N the number of DOFs), the observed order r of three
successive meshes solves (q1 − q2)/(q2 − q3) = (h1^r − h2^r)/(h2^r − h3^r) (local orders, noisy on
unstructured meshes), and the order and the extrapolated value (Richardson) come from a least-squares fit
q ≈ q∞ + C h^r over all meshes. Expected orders for elements of
order p on smooth solutions: eigenfrequencies 2p, energy norm p, L2 norm and point values p + 1 (re-entrant
corners and clamped edges lower them).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from loudpy.errors import ModelError, StateError

__all__ = ["Adaptive", "convergence", "Convergence", "estimate_errors"]

AIM = 0.5          # the sizes aim at AIM × target (the number of cells grows after the update: aiming at the
                   # target itself leaves the next estimate above it, the loop then creeps)


# ════════════════════════════════════════════════════════════════════════════
#  Error estimate of results
# ════════════════════════════════════════════════════════════════════════════
def _group_spaces(mesh, groups, physics):
    """[(group name, Space, D or None)] of the 2D groups of `mesh` estimated for `physics` ("solid" / "air")."""
    from loudpy.fem.reference import element_for_cells
    from loudpy.fem.space import Space
    from loudpy.results.derived import elasticity
    out = []
    for name, g in mesh.groups.items():
        if g.dim != 2 or not len(g):
            continue
        d = groups.get(name, {})
        if physics == "solid" and d.get("role") == "solid":
            if "E" not in d or "nu" not in d:
                raise ModelError(f"no elastic data (E, ν) for the mesh group {name!r}")
            D = elasticity(d["E"], d["nu"])
        elif physics == "air" and d.get("role") == "air":
            D = None
        else:
            continue
        cells = np.asarray(g.cells)
        out.append((name, Space(mesh.node_tags, mesh.coords, cells, element_for_cells(2, cells.shape[1]),
                                2 if physics == "solid" else 1), D))
    return out


def estimate_errors(res) -> dict:
    """
    The relative error estimate of axisymmetric results (module docstring):
        {"estimate": largest relative estimate, "by_physics": {"solid": …, "air": …} (largest over entries),
         "cells": [(physics, Space, relative η² per cell (largest over entries))]}
    Needs the physical data of the mesh groups (results of a study, or read from a format-3 file).
    """
    from loudpy.fem import estimate as est
    if res.model is None or not res.model.groups:
        raise StateError("the error estimate needs the physical data of the mesh groups (format-3 results)")
    groups = res.model.groups
    by_physics, cells = {}, []
    for b in res.blocks:
        for physics, name in (("solid", "u"), ("air", "p")):
            if name not in b.fields:
                continue
            mesh = b.mesh_of(name)
            arr = np.asarray(b.fields[name])
            if physics == "solid" and (arr.ndim != 3 or arr.shape[2] != 2):
                raise ModelError("lp.Adaptive estimates axisymmetric displacements (m = 0, components r, z)")
            spaces = _group_spaces(mesh, groups, physics)
            if not spaces:
                continue
            rows = [mesh._rows(sp.tags) for _, sp, _ in spaces]
            worst = [np.zeros(sum(len(c) for c in sp.cell_nodes)) for _, sp, _ in spaces]
            for k in range(arr.shape[0]):
                eta2, norm2 = [], 0.0
                for (_, sp, D), r in zip(spaces, rows):
                    V = arr[k][r]
                    if physics == "solid":
                        eta2.append(est.elastic_indicator(sp, V.ravel(), D))
                        norm2 += est.elastic_energy(sp, V.ravel(), D).sum()
                    else:
                        eta2.append(est.recovery_indicator(sp, V, axisymmetric=True))
                        norm2 += est.gradient_energy(sp, V, axisymmetric=True).sum()
                if not norm2 > 0:
                    continue
                total = np.sqrt(sum(e.sum() for e in eta2) / norm2)
                by_physics[physics] = max(by_physics.get(physics, 0.0), float(total))
                for w, e in zip(worst, eta2):
                    np.maximum(w, e / norm2, out=w)
            cells += [(physics, sp, w) for (_, sp, _), w in zip(spaces, worst)]
    if not by_physics:
        raise ModelError("nothing to estimate: no solid displacement or air pressure in the results")
    return {"estimate": max(by_physics.values()), "by_physics": by_physics, "cells": cells}


def _n_dofs(res) -> int:
    b = res.blocks[0]
    return int(sum(np.prod(np.shape(a)[1:]) for a in b.fields.values()))


def _triangles(corners):
    """Corner rows (n, 3 | 4) as triangles (quadrilaterals split in two)."""
    return corners if corners.shape[1] == 3 else np.vstack([corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]])


def size_view(estimate: dict, target: float, path) -> Path:
    """
    New sizes for the relative `target` from estimate_errors(res), written as a gmsh size view at `path`.
    Sizes by equidistribution per physics; per node (by node tag, over every group and physics) the smallest
    size of the cells around it: a node shared by a solid and the air (or two solids) has one size, the finer,
    so the interface curves are meshed at it.
    """
    from loudpy.fem import adapt
    from loudpy.fem.estimate import cell_sizes
    parts = []                                                   # (tags of the corners, new sizes, space)
    for physics in ("solid", "air"):
        ph = [(sp, w) for name, sp, w in estimate["cells"] if name == physics]
        if not ph:
            continue
        eta2 = np.concatenate([w for _, w in ph])
        h = np.concatenate([cell_sizes(sp) for sp, _ in ph])
        hn = adapt.new_sizes(h, eta2, target, order=ph[0][0].element.order)
        i = 0
        for sp, w in ph:
            nv = len(sp.element.cell.vertices)
            parts.append((sp.tags[np.vstack(sp.cell_nodes)[:, :nv]], hn[i:i + len(w)], sp))
            i += len(w)
    tags = np.unique(np.concatenate([sp.tags for _, _, sp in parts]))
    coords = np.zeros((len(tags), 2))
    for _, _, sp in parts:
        coords[np.searchsorted(tags, sp.tags)] = sp.node_coords
    hnode = np.full(len(tags), np.inf)
    tris = []
    for ct, hc, _ in parts:
        rows = np.searchsorted(tags, ct)
        np.minimum.at(hnode, rows.ravel(), np.repeat(hc, rows.shape[1]))
        tris.append(_triangles(rows))
    return adapt.write_pos(path, coords, np.vstack(tris), hnode)


# ════════════════════════════════════════════════════════════════════════════
#  Adaptive
# ════════════════════════════════════════════════════════════════════════════
def _variant(study, model):
    st = copy.copy(study)
    st.model, st.results = model, None
    return st


def _check_study(study, what):
    from loudpy.study.base import Study
    if not isinstance(study, Study):
        raise ModelError(f"{what} takes a study (lp.Modes, lp.FrequencyResponse …), got {type(study).__name__}")
    if getattr(study, "is_scene", False):
        raise NotImplementedError(f"{what}: 2D models only (a scene's meshes are not adapted)")
    if study.model.meshed is None or study.model._mesh_path is None:
        raise StateError("mesh the model first: model.mesh(path)")


class Adaptive:
    """Solves `study` on meshes adapted to its own error estimate until the relative estimate is ≤ target
    (module docstring)."""

    def __init__(self, study, target: float = 0.05, max_iterations: int = 4, folder=None):
        from loudpy.study.transient import Transient
        _check_study(study, "lp.Adaptive")
        if isinstance(study, Transient):
            raise NotImplementedError("lp.Adaptive: lp.Modes and lp.FrequencyResponse (not lp.Transient)")
        if not 0 < target < 1:
            raise ModelError(f"lp.Adaptive target is a relative error in (0, 1), got {target!r}")
        if int(max_iterations) < 0:
            raise ModelError("lp.Adaptive max_iterations must be ≥ 0")
        self.study, self.target, self.max_iterations = study, float(target), int(max_iterations)
        self.folder = Path(folder) if folder is not None else None
        self.history, self.results, self.model = [], None, None

    def solve(self, *args, **kwargs):
        """Calls ``study.solve(*args, **kwargs)`` on each mesh; returns the results of the last one."""
        if kwargs.get("remesh") is not None:
            raise ModelError("lp.Adaptive adapts the mesh itself: no remesh= (lp.PerWavelength) inside it")
        base = self.study.model
        folder = self.folder or base._mesh_path.parent
        stem = base._mesh_path.stem
        model, self.history = base, []
        for it in range(self.max_iterations + 1):
            st = _variant(self.study, model)
            res = st.solve(*args, **kwargs)
            e = estimate_errors(res)
            self.history.append({"iteration": it, "mesh": str(model._mesh_path), "n_dofs": _n_dofs(res),
                                 "n_cells": int(sum(len(sp.cell_nodes[0]) for _, sp, _ in e["cells"])),
                                 "estimate": e["estimate"], **{f"estimate_{k}": v for k, v in e["by_physics"].items()}})
            if e["estimate"] <= self.target or it == self.max_iterations:
                break
            pos = size_view(e, AIM * self.target, folder / f"{stem}_adapt{it + 1}.pos")
            model = base.resolved()
            model.mesh(folder / f"{stem}_adapt{it + 1}.msh", size_field=pos)
        res.meta["adaptive"] = json.dumps({"target": self.target, "history": self.history})
        self.results, self.model = res, model
        return res

    @property
    def converged(self) -> bool:
        """True when the last estimate is at or below the target."""
        return bool(self.history) and self.history[-1]["estimate"] <= self.target

    def __repr__(self):
        head = f"lp.Adaptive(target {self.target:g}, {len(self.history)} mesh(es)"
        if not self.history:
            return head + ", not solved)"
        lines = [head + (", converged)" if self.converged else ", not converged)"),
                 f"  {'mesh':>4} {'DOFs':>8} {'cells':>7} {'estimate':>9} {'solid':>9} {'air':>9}"]
        for h in self.history:
            f = [f"{h.get(k, float('nan')):9.3g}" for k in ("estimate", "estimate_solid", "estimate_air")]
            lines.append(f"  {h['iteration']:>4} {h['n_dofs']:>8} {h['n_cells']:>7} " + " ".join(f))
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  Convergence
# ════════════════════════════════════════════════════════════════════════════
def _scaled(model, factor):
    """A resolved copy of the model with every mesh size (regions, boundaries, mesh controls) × factor."""
    m = model.resolved()
    for it in m.regions + m.boundaries:
        if getattr(it, "mesh_size", None) is not None:
            it.mesh_size = it.mesh_size * factor
    for c in m.mesh_controls:
        c.size = c.size * factor
    return m


def _order(h, q):
    """Observed order of three meshes (h decreasing) for each column of q (3, m); NaN when not monotone."""
    from scipy.optimize import brentq
    out = np.full(q.shape[1], np.nan)
    for j in range(q.shape[1]):
        d1, d2 = q[0, j] - q[1, j], q[1, j] - q[2, j]
        if d2 == 0 or d1 / d2 <= 0:
            continue
        ratio = d1 / d2

        def f(r):
            return (h[0] ** r - h[1] ** r) / (h[1] ** r - h[2] ** r) - ratio
        try:
            out[j] = brentq(f, 0.05, 30.0)
        except ValueError:
            pass
    return out


def _fit(h, q):
    """Least-squares fit q ≈ q∞ + C h^r over all meshes, per column: (q∞, r); r searched in [0.3, 12]."""
    from scipy.optimize import minimize_scalar
    q_inf, order = np.full(q.shape[1], np.nan), np.full(q.shape[1], np.nan)
    for j in range(q.shape[1]):
        y = q[:, j]

        def solve(r):
            A = np.column_stack([np.ones_like(h), h ** r])
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            return c, float(np.sum((A @ c - y) ** 2))
        grid = np.linspace(0.3, 12.0, 118)
        r0 = grid[int(np.argmin([solve(r)[1] for r in grid]))]
        r = minimize_scalar(lambda r: solve(r)[1], bounds=(max(0.3, r0 - 0.1), min(12.0, r0 + 0.1)),
                            method="bounded").x
        q_inf[j], order[j] = solve(r)[0][0], r
    return q_inf, order


class Convergence:
    """
    The table of lp.convergence: factors, n_dofs, h = N^(-1/2), values (n_meshes, m),
    order and extrapolated (m,): least-squares fit q ≈ q∞ + C h^r over all meshes (three or more),
    local_orders (n_meshes − 2, m): the order of each three successive meshes (noisy on unstructured meshes),
    errors: |q − q∞| / |q∞| per mesh, expected: the orders of smooth solutions for the model's element order.
    Complex quantities: tabulated, not fitted (give abs / angle / real parts as the quantity).
    """

    def __init__(self, factors, n_dofs, values, mesh_order, results):
        self.factors = np.asarray(factors, float)
        self.n_dofs = np.asarray(n_dofs, int)
        self.h = self.n_dofs.astype(float) ** -0.5
        self.values = np.asarray(values)
        self.results = results
        q = self.values
        m = q.shape[1]
        if len(self.factors) >= 3 and not np.iscomplexobj(q):
            self.local_orders = np.array([_order(self.h[i:i + 3], q[i:i + 3]) for i in range(len(q) - 2)])
            self.extrapolated, self.order = _fit(self.h, q)
        else:
            self.local_orders = np.zeros((0, m))
            self.extrapolated, self.order = np.full(m, np.nan), np.full(m, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.errors = np.abs(self.values - self.extrapolated) / np.abs(self.extrapolated)
        p = int(mesh_order)
        self.expected = {"eigenfrequency": 2 * p, "energy norm": p, "L2 norm / point values": p + 1}

    def __repr__(self):
        m = self.values.shape[1]
        cols = min(m, 6)
        lines = [f"lp.convergence: {len(self.factors)} meshes, {m} value(s)" + (f" (first {cols} shown)" if m > cols else ""),
                 "  factor     DOFs   " + " ".join(f"{'q' + str(j):>12}" for j in range(cols))]
        for i, (f, n) in enumerate(zip(self.factors, self.n_dofs)):
            lines.append(f"  {f:6.3g} {n:8d}   " + " ".join(f"{self.values[i, j]:12.6g}" for j in range(cols)))
        for i, o in enumerate(self.local_orders):
            lines.append(f"  order {i}-{i + 2}      " + " ".join(f"{o[j]:12.3g}" for j in range(cols)))
        if len(self.local_orders):
            lines.append("  order (fit)      " + " ".join(f"{self.order[j]:12.3g}" for j in range(cols)))
            lines.append("  extrapolated     " + " ".join(f"{self.extrapolated[j]:12.6g}" for j in range(cols)))
        lines.append("  expected orders (in h, smooth solutions): " +
                     ", ".join(f"{k} {v}" for k, v in self.expected.items()))
        return "\n".join(lines)


def convergence(study, factors=(1.0, 0.7, 0.5), quantity=None, *args, folder=None, **kwargs) -> Convergence:
    """
    Solves `study` on meshes with all sizes × each factor and tabulates quantity(results) (module docstring).
    Extra arguments go to study.solve. quantity default: the mode frequencies (lp.Modes, the number of modes
    common to all meshes).
    """
    from loudpy.study.modes import Modes
    _check_study(study, "lp.convergence")
    factors = [float(f) for f in factors]
    if len(factors) < 2 or any(f <= 0 for f in factors) or any(b >= a for a, b in zip(factors, factors[1:])):
        raise ModelError("lp.convergence factors: at least two, positive, decreasing (finer meshes)")
    if quantity is None:
        if not isinstance(study, Modes):
            raise ModelError("lp.convergence needs quantity=(results → values) for this study")
    base = study.model
    folder = Path(folder) if folder is not None else base._mesh_path.parent
    stem = base._mesh_path.stem
    values, n_dofs, results = [], [], []
    for f in factors:
        m = _scaled(base, f)
        m.mesh(folder / f"{stem}_h{f:g}.msh")
        res = _variant(study, m).solve(*args, **kwargs)
        results.append(res)
        n_dofs.append(_n_dofs(res))
        values.append(np.atleast_1d(np.asarray(res.values if quantity is None else quantity(res))))
    n = min(len(v) for v in values)
    if quantity is None:
        values = [v[:n] for v in values]
    elif any(len(v) != n for v in values):
        raise ModelError("lp.convergence: quantity(results) must give the same number of values on every mesh")
    return Convergence(factors, n_dofs, np.vstack(values), base.mesh_order, results)
