"""
Modes: eigenmodes of a model.

    modes = lp.Modes(model).solve(f_max=6000)                        structural modes (in vacuo)
    modes = lp.Modes(model).solve(f_max=6000, orders=[0, 1, 2, 3])   azimuthal orders m (3D modes of
                                                                     the axisymmetric structure)
    modes = lp.Modes(model, physics="coupled").solve(f_max=9000, dtn_center=..., ...)
                                                                     structure + air with the exact
                                                                     DtN radiation (lp.Radiation)
    modes = lp.Modes(model, physics="acoustic").solve(f_max=3000)    air only (rigid structure)

Structural modes use several shift-invert solves spread over the band (method="multishift",
default) or one solve with n_modes around f_target (method="arpack").

Damping is read from the assembled operators:

    K real,    no C   GEP (K − λM)φ = 0, real λ = ω²
    K real,    C      QEP (s²M + sC + K)φ = 0 (state space)
    K complex, no C   GEP with complex λ (hysteretic)
    K complex, C      QEP

and ζ = −Re(s)/|s|, η = 2ζ from the eigenvalue s for all four.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import warnings

import numpy as np

from loudpy.checks import check
from loudpy.physics.registry import select
from loudpy.results.results import Block, ModeResults
from loudpy.solvers import eigen as eig
from loudpy.study.base import Study
from loudpy.errors import ModelError

log = logging.getLogger("loudpy.study")

PHYSICS = ("structure", "coupled", "acoustic", "walls")


def _damping(K, C):
    """(K, M-compatible C or None, hysteretic, viscous) from assembled operators."""
    if C is not None and np.abs(C).sum() < 1e-30:
        C = None
    return C, eig.has_imaginary_part(K), C is not None


class Modes(Study):
    """Eigenmodes of a Model (physics "structure", "acoustic" or "coupled") or of an lp.Scene
    (coupled driver + cavity [+ wall] modes; physics="walls": the in-vacuo modes of its elastic walls).

        modes = lp.Modes(model).solve(f_max=20000)
        walls = lp.Modes(scene, physics="walls").solve(f_max=2000)"""
    accepts_scene = True

    def __init__(self, model, physics: str = "structure"):
        super().__init__(model)
        if physics not in PHYSICS:
            raise ModelError(f"physics must be one of {PHYSICS}, got {physics!r}")
        self.physics = physics

    def solve(self, f_max: float, f_min: float = 20.0, *, orders=None, method: str = "multishift",
              n_per_shift: int = 20, n_modes: int = 30, f_target: float = 100.0, output=None,
              **options) -> ModeResults:
        """
        f_min, f_max : band of the modes [Hz]
        output       : an lp.Output (fields kept, every n-th mode, probes over the modes); 2D models
        orders       : azimuthal orders m (structure only; None: the axisymmetric modes, m = 0)
        method       : "multishift" (shifts over the band, n_per_shift modes each) or "arpack"
                       (n_modes around f_target)
        options      : multishift: f_targets, dedup_freq_tol_hz, mac_tol;
                       coupled: dtn_center, dtn_half, dtn_order, mac_tol, cover_zeta_max, cover_overlap,
                       max_shifts, cover_goal, cover_avoid_frac, dedup_freq_tol_hz, dedup_rel_tol,
                       checkpoint_path, checkpoint_every (see coupled_modes)
        """
        check(f_max, "frequency_positive", "Modes.solve(f_max)")
        check(f_min, "frequency", "Modes.solve(f_min)")
        if f_min >= f_max:
            raise ModelError(f"Modes.solve: f_min = {f_min:g} Hz must be below f_max = {f_max:g} Hz")
        if int(n_per_shift) < 1 or int(n_modes) < 1 or not f_target > 0:
            raise ModelError(f"Modes.solve: n_per_shift ≥ 1, n_modes ≥ 1 and f_target > 0, got {n_per_shift!r}, "
                             f"{n_modes!r} and {f_target!r}")
        if self.is_scene:                       # drivers + cavity modes (scene configuration "cavity")
            if output is not None:
                raise ModelError("output= is for 2D models (scene results keep modal coordinates only)")
            from loudpy.study import scene_response
            if self.physics == "walls":
                self.results = self._finish(scene_response.wall_modes(self, f_max=f_max))
            else:
                self.results = self._finish(scene_response.coupled_modes(self, f_max=f_max))
            return self.results
        if self.physics == "walls":
            raise ModelError('physics="walls": the elastic walls of an lp.Scene (scene.walls(…))')
        parts = self._parts()
        if self.physics == "structure":
            if orders is not None:
                blocks = [fourier_modes(parts, m, f_min, f_max, n_per_shift, **options) for m in orders]
            elif method == "multishift":
                blocks = [structural_modes(parts, f_min, f_max, n_per_shift=n_per_shift, **options)]
            elif method == "arpack":
                blocks = [structural_modes(parts, f_min, f_max, n_modes=n_modes, f_target=f_target)]
            else:
                raise ModelError(f"method must be 'multishift' or 'arpack', got {method!r}")
        elif self.physics == "coupled":
            blocks = [coupled_modes(parts, f_min, f_max, n_per_shift, boundaries=self.model.boundary_nodes(),
                                    meta=self._meta(), **options)]
        else:
            blocks = [acoustic_modes(parts, n_modes, f_min, f_max, f_target)]
        blocks.sort(key=lambda b: (0, b.meta["order"]) if "order" in b.meta else (1, b.label))
        from loudpy.results.output import apply
        self.results = self._finish(apply(ModeResults(blocks, self._meta(), self.model.boundary_nodes()), output))
        return self.results

    def _meta(self):
        return {"study": "Modes", "physics": self.physics, "geometry": str(self.model.geometry)}


# ════════════════════════════════════════════════════════════════════════════
#  Structural modes
# ════════════════════════════════════════════════════════════════════════════
def _structural_block(s, parts, label, order, s_vals, phi, K, M, C, hysteretic, viscous, f_min, f_max, params):
    from loudpy.physics.systems import meshes, nodal
    frequencies, zeta, loss, s_vals, phi = eig.filter_modes(s_vals, phi, f_min, f_max,
                                                            filter_halfplane=(hysteretic or viscous))
    reduced = eig.normalise_modes(phi, s_vals, M, C, hysteretic, viscous)
    full = np.array([s.meca.expand(reduced[:, i]) for i in range(reduced.shape[1])])
    eig.print_mode_summary(frequencies, zeta, loss, hysteretic, viscous)
    dpn = int(s.meca.dofs_per_node)
    meta = {k: v for k, v in params.items() if np.isscalar(v)}
    meta["order"] = order
    fields = {"u": nodal(full.reshape(len(frequencies), -1), len(s.meca.unique_tags), dpn)}
    return Block(meshes(s, parts), np.asarray(frequencies), fields, {"zeta": np.asarray(zeta)}, meta, label)


def structural_modes(parts, f_min, f_max, *, n_per_shift=None, n_modes=None, f_target=100.0, f_targets=None,
                     dedup_freq_tol_hz=0.5, mac_tol=0.99) -> Block:
    """Axisymmetric structural modes (m = 0): multishift (n_per_shift) or one ARPACK solve (n_modes)."""
    from loudpy.physics.systems import assemble
    s = assemble(parts)
    K, M = eig.to_csc_complex(s.K, s.M)
    [C] = eig.to_csc_complex(s.C)
    C, hysteretic, viscous = _damping(K, C)
    log.info("structure: K %s, viscous damping %s", "complex" if hysteretic else "real", "yes" if viscous else "no")
    if n_modes is not None:                                       # one ARPACK solve around f_target
        s_vals, phi = (eig.solve_qep(K, M, C, n_modes, f_target) if viscous
                       else eig.solve_gep(K, M, n_modes, f_target, hysteretic))
        s_vals, phi = eig.mac_deduplicate(s_vals, phi, freq_tol_rad=1.0, mac_tol=0.99999)
        params = dict(n_modes=n_modes, f_min=f_min, f_max=f_max, f_target=f_target)
    else:
        f_targets = f_targets if f_targets is not None else eig.default_shifts(f_min, f_max)
        s_vals, phi = eig.multishift(K, M, C, f_targets, n_per_shift, hysteretic)
        s_vals, phi = eig.mac_deduplicate(s_vals, phi, freq_tol_rad=2.0 * np.pi * dedup_freq_tol_hz,
                                          mac_tol=mac_tol)
        params = dict(n_modes_per_shift=n_per_shift, f_min=f_min, f_max=f_max)
    return _structural_block(s, parts, "m0", 0, s_vals, phi, K, M, C, hysteretic, viscous, f_min, f_max, params)


def fourier_modes(parts, m, f_min, f_max, n_per_shift=20, f_targets=None, dedup_freq_tol_hz=0.5,
                  mac_tol=0.99) -> Block:
    """
    In-vacuo modes of azimuthal order m of the axisymmetric structure (multishift).

    The 3D mode is  u_r = U_r(r,z) cos mθ,  u_z = U_z(r,z) cos mθ,  u_θ = U_θ(r,z) sin mθ
    (each mode of order m ≥ 1 has a twin rotated by π/(2m): same frequency). m = 0 gives the
    axisymmetric modes (2 DOF per node); m ≥ 1 has 3 DOF per node (U_r, U_z, U_θ). The modes are
    normalised over the full 3D body (θ-integral π for m ≥ 1, 2π for m = 0).
    """
    from loudpy.physics.systems import assemble_fourier
    m = int(m)
    if m == 0:
        return structural_modes(parts, f_min, f_max, n_per_shift=n_per_shift, f_targets=f_targets,
                                dedup_freq_tol_hz=dedup_freq_tol_hz, mac_tol=mac_tol)
    s = assemble_fourier(parts, m)
    K, M = eig.to_csc_complex(s.K, s.M)
    [C] = eig.to_csc_complex(s.C)
    C, hysteretic, viscous = _damping(K, C)
    log.info("azimuthal order m = %d: %d DOF (3 per node), %d after constraints", m, s.meca.matrix_size, K.shape[0])
    f_targets = f_targets if f_targets is not None else eig.default_shifts(f_min, f_max)
    s_vals, phi = eig.multishift(K, M, C, f_targets, n_per_shift, hysteretic)
    s_vals, phi = eig.mac_deduplicate(s_vals, phi, freq_tol_rad=2.0 * np.pi * dedup_freq_tol_hz, mac_tol=mac_tol)
    params = dict(azimuthal_order=m, n_modes_per_shift=n_per_shift, f_min=f_min, f_max=f_max)
    return _structural_block(s, parts, f"m{m}", m, s_vals, phi, K, M, C, hysteretic, viscous, f_min, f_max,
                             params)


# ════════════════════════════════════════════════════════════════════════════
#  Acoustic modes (rigid structure)
# ════════════════════════════════════════════════════════════════════════════
def acoustic_modes(parts, n_modes=30, f_min=20.0, f_max=20_000.0, f_target=100.0) -> Block:
    """Modes of the air alone, (H − ω²Q) p = 0 (real symmetric), mass-normalised pᵀQp = 1."""
    from scipy.sparse.linalg import eigsh

    from loudpy.physics.systems import assemble_acoustic, meshes, nodal
    s = assemble_acoustic(parts)
    assert s.H is not None and s.Q is not None
    H, Q = s.H.tocsc().astype(float), s.Q.tocsc().astype(float)
    n_dof = H.shape[0]
    sigma = float((2.0 * np.pi * f_target) ** 2)
    lam, phi = eigsh(H, M=Q, k=min(n_modes, n_dof - 2), sigma=sigma, which="LM", tol=1e-10,
                     v0=eig.start_vector(n_dof, complex_=False))
    freqs = np.sqrt(np.maximum(lam, 0.0)) / (2.0 * np.pi)
    keep = (freqs >= f_min) & (freqs <= f_max)
    freqs, phi = freqs[keep], phi[:, keep]
    order = np.argsort(freqs)
    freqs, phi = freqs[order], phi[:, order]
    shapes = np.ascontiguousarray(phi.T)
    for k in range(len(freqs)):
        norm = shapes[k] @ (Q @ shapes[k])
        if norm > 1e-30:
            shapes[k] /= np.sqrt(norm)
    eig.print_acoustic_summary(freqs)
    meta = dict(n_modes=n_modes, f_min=f_min, f_max=f_max, f_target=f_target)
    return Block(meshes(s, parts), freqs, {"p": nodal(shapes, len(s.acou.unique_tags), 1)}, {}, meta, "acou")


# ════════════════════════════════════════════════════════════════════════════
#  Coupled structure + air modes, exact DtN radiation
# ════════════════════════════════════════════════════════════════════════════
def _goal_check(s, parts, Ks, Ms, Cs, nm, na, f_min, f_max, goal):
    """Goal-oriented stop of the covering: modal sum vs exact pressure at receivers (DtN form)."""
    from loudpy.solvers.linear import factorize
    F = s.F * float(goal.get("force", 1.0))
    rhs = np.zeros(Ks.shape[0], dtype=complex)
    rhs[:nm] = F
    xy = {}
    for o in parts:
        cc = np.asarray(o.node_coords).reshape(len(o.node_tags), -1)
        for t, c_ in zip(o.node_tags, cc):
            xy[int(t)] = c_[:2]
    axy = np.array([xy[int(t)] for t in s.acou.unique_tags])
    rec = np.array([int(np.argmin(np.sum((axy - np.asarray(q)) ** 2, axis=1))) for q in goal["receivers"]])
    freqs = np.asarray(goal.get("freqs", np.geomspace(max(20.0, f_min), f_max / 1.6, 30)))
    tol = float(goal.get("tol_db", 1.0))
    p_ex = np.empty((freqs.size, rec.size), dtype=complex)
    for i, f in enumerate(freqs):
        sv = 2j * np.pi * f
        x = factorize(sv * sv * Ms + sv * Cs + Ks).solve(rhs)
        p_ex[i] = -sv * x[nm:nm + na][rec]
    log.info("goal: exact pressure at %d receivers, %d frequencies (%.0f–%.0f Hz), tolerance %g dB",
             rec.size, freqs.size, freqs.min(), freqs.max(), tol)
    jw = 2j * np.pi * freqs[:, None]
    cache = {}

    def stop_fn(s_list, phi_list):
        for k in range(len(cache), len(s_list)):
            (sk,), (real,) = eig.real_modes([s_list[k]])
            x = phi_list[k]
            if not ((sk.real < 0) and (real or sk.imag > 0)):
                cache[k] = None
                continue
            x = x / np.sqrt(x @ (2.0 * sk * (Ms @ x) + Cs @ x))
            cache[k] = (sk, (0.5 if real else 1.0) * (x[:nm] @ F), -sk * x[nm:nm + na][rec])
        items = [v for v in cache.values() if v is not None]
        if not items:
            return False, "no mode yet"
        S = np.array([v[0] for v in items])
        A = np.array([v[1] for v in items])
        P = np.array([v[2] for v in items])
        p = (((jw / S) * A / (jw - S)) @ P
             + ((jw / np.conj(S)) * np.conj(A) / (jw - np.conj(S))) @ np.conj(P))
        e = 20 * np.log10(np.abs(p) / np.abs(p_ex))
        emax = float(np.abs(e).max())
        return emax <= tol, f"max error {emax:.2f} dB, rms {np.sqrt(np.mean(e ** 2)):.2f} dB ({len(items)} modes)"
    return stop_fn


def coupled_modes(parts, f_min=20.0, f_max=5000.0, n_per_shift=30, *, dedup_freq_tol_hz=0.5,
                  dedup_rel_tol=0.002, mac_tol=0.99, cover_zeta_max=0.999, cover_overlap=0.85, max_shifts=60,
                  dtn_center=None, dtn_half=True, dtn_order=None, cover_goal=None, cover_avoid_frac=0.25,
                  checkpoint_path=None, checkpoint_every=3, boundaries=None, meta=None) -> Block:
    """
    Coupled structure–air modes of the loudspeaker radiating in free space, with the exact DtN
    condition on the radiation arc(s) (lp.Radiation; one arc per half-space, oriented ±1).

    dtn_center : centre (0, z0) of every arc, {arc name: (0, z0)}, or None (found from each arc)
    dtn_half   : hemispheres on a rigid baffle through their centres (even orders), else spheres
    dtn_order  : truncation N (default ⌈k_max R⌉ + 10)

    The modes come from a gap-free covering of the s-plane (solvers.eigen.covering): every mode
    with f in [f_min, f_max] and ζ ≤ cover_zeta_max, including the heavily damped radiation
    modes the modal sum of the pressure needs, and the real (overdamped) eigenvalues, stored with
    Im(s) = 0. A real mode enters the modal sum once:
        p(jω) = Σ_oscillating [(jω/s)p a/(jω − s) + conjugate] + Σ_real (jω/s)p a/(jω − s)

    cover_avoid_frac : Rayleigh damping sends the structural modes above 1/(πβ) to ζ > 1, piling
                       them on the real axis at s = −1/β; the covering avoids a disc of radius
                       cover_avoid_frac/β around each pile (0: off).
    cover_goal       : stop once the modal sum matches the exact forced pressure within tol_db at
                       given receivers: dict(receivers=[(r, z), …], freqs=…, tol_db=1.0, force=1.0)
    checkpoint_path  : the modes found so far are written there (format 2, atomic) every
                       checkpoint_every shifts

    Normalisation x_kᵀ(2 s_k Ms + Cs) x_k = 1 on the symmetric u–ψ–φ form (p = −sψ), so that for a
    structural force F:  u(jω) = Σ_k u_k (u_kᵀF)/(jω − s_k) + conj,  p(jω) = Σ_k (jω/s_k) p_k (u_kᵀF)/(jω − s_k) + conj.
    The block holds "u", "p", "zeta" and the eigenvalues "s".
    """
    from loudpy.physics.systems import assemble, dtn_system, meshes, nodal
    if not select(parts, "radiation"):
        raise ModelError("coupled modes need a radiation boundary (lp.Radiation)")
    s = assemble(parts, air=True)
    Ks, Ms, Cs, ds = dtn_system(s, parts, dtn_center, f_max, dtn_half, dtn_order)
    nm, na = ds["nm"], ds["na"]
    if np.abs(Cs).sum() == 0:
        raise ModelError("the coupled DtN system has no damping matrix (C = 0)")
    stop_fn = _goal_check(s, parts, Ks, Ms, Cs, nm, na, f_min, f_max, cover_goal) if cover_goal is not None else None
    avoid = []
    if cover_avoid_frac > 0:
        betas = sorted({float(getattr(o, "beta_ray", 0.0) or 0.0) for o in parts} - {0.0})
        for b in betas:
            if 1.0 / b <= 1.3 * 2 * np.pi * f_max:
                avoid.append((-1.0 / b + 0j, cover_avoid_frac / b))
    ms = meshes(s, parts)

    def block(s_raw, phi_raw, shifts, quiet=False):
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()) if quiet \
                else contextlib.nullcontext():
            if quiet:
                warnings.simplefilter("ignore")
            freqs, zeta, loss, s_vals, phi = eig.filter_modes(s_raw, phi_raw, f_min, f_max, filter_halfplane=True,
                                                              keep_real=True)
        phi = eig.normalise_symmetric(phi, s_vals, Ms, Cs)
        u = np.array([s.meca.expand(phi[:nm, k]) for k in range(phi.shape[1])])
        p = np.ascontiguousarray((-s_vals[None, :] * phi[nm:nm + na]).T)
        if not quiet:
            eig.print_mode_summary(freqs, zeta, loss, True, True)
        params = dict(n_modes_per_shift=n_per_shift, f_min=f_min, f_max=f_max, cover_zeta_max=cover_zeta_max,
                      exterior="dtn", dtn_arcs=", ".join(a["name"] for a in ds["arcs"]),
                      dtn_orders=len(ds["arcs"][0]["orders"]), dtn_half=dtn_half, n_shifts=len(shifts))
        fields = {"u": nodal(u.reshape(len(freqs), -1), len(s.meca.unique_tags), 2),
                  "p": nodal(p.reshape(len(freqs), -1), len(s.acou.unique_tags), 1)}
        return Block(ms, np.asarray(freqs), fields, {"zeta": np.asarray(zeta), "s": np.asarray(s_vals)}, params,
                     "fsi")

    def checkpoint(s_list, phi_list, shifts):
        b = block(np.array(s_list), np.column_stack(phi_list), shifts, quiet=True)
        tmp = f"{checkpoint_path}.partial"
        try:
            ModeResults([b], dict(meta or {}, checkpoint_shifts=len(shifts)), boundaries).save(tmp)
            os.replace(tmp, checkpoint_path)
            log.info("checkpoint: %d modes after %d shifts → %s", len(b), len(shifts), checkpoint_path)
        except OSError as e:                                       # e.g. a synced folder holding the file
            log.warning("checkpoint skipped (%s: %s)", type(e).__name__, e)

    s_vals, phi, shifts = eig.covering(
        Ks, Ms, Cs, f_min, f_max, n_per_shift, zeta_cover=cover_zeta_max, overlap=cover_overlap,
        max_shifts=max_shifts, freq_tol_rad=2.0 * np.pi * dedup_freq_tol_hz, mac_tol=mac_tol,
        rel_tol=dedup_rel_tol, stop_fn=stop_fn, avoid=avoid,
        progress_fn=checkpoint if checkpoint_path else None, progress_every=checkpoint_every)
    return block(s_vals, phi, shifts)
