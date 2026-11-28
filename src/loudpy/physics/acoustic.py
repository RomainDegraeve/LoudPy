"""
Acoustic operators of the air regions (roles "air" and "pml"): H and Q, 1 DOF per node.

A PML region (role "pml") carries its layer geometry in `part.layer` (geometry.naming.PMLLayer)
and its absorption parameters (alpha, n, omega_pml); its H and Q are complex.
"""
from __future__ import annotations

import numpy as np

from loudpy.fem.assembly import ElementAssembler
from loudpy.fem.kernels import (diffusion_diag_kernel, laplace_kernel, mass_kernel, coefficient,
                                quadrature_points)
from loudpy.physics.registry import select


def _stretch(x, x0, t, sign, alpha, n, lambda0):
    """PML complex stretching along one coordinate (as in commercial FE software): s(x) and the stretched x̃."""
    xi = sign * (x - x0) / t
    inside = xi > 0.0
    xi = np.clip(xi, 0.0, 1.0)
    xi_nm1 = np.where(inside, xi ** (n - 1), 0.0)
    xi_n = xi_nm1 * xi
    s = np.where(inside, 1.0 - 1j * alpha * lambda0 * (n / t) * xi_nm1, 1.0 + 0j)
    x_tilde = np.where(inside, x - 1j * alpha * lambda0 * xi_n * sign, x + 0j)
    return s, x_tilde


def pml_coefficients(layer, x, alpha, n, omega, c):
    """
    s_r, s_z and r̃ at points x (…, 2) of a PML layer (geometry.naming.PMLLayer). The stretched
    Helmholtz operator is ∫ (s_z/s_r ∂_r p ∂_r q + s_r/s_z ∂_z p ∂_z q) 2π r̃ dA and the mass
    ∫ p q s_r s_z 2π r̃ / c² dA (s never takes the sign: Im s < 0).
    """
    lambda0 = 2.0 * np.pi * c / omega
    r, z = x[..., 0], x[..., 1]
    sr = np.ones(r.shape, complex)
    sz = np.ones(r.shape, complex)
    r_t = r + 0j
    if layer.use_r:
        sr, r_t = _stretch(r, layer.r0, layer.tr, layer.sgn_r, alpha, n, lambda0)
    if layer.use_z:
        sz, _ = _stretch(z, layer.z0, layer.tz, layer.sgn_z, alpha, n, lambda0)
    return sr, sz, r_t


class AcouAssembler(ElementAssembler):
    """Assembler of the air operators H = ∫∇N·∇N 2πr and Q = ∫NN/c² 2πr (with the PML stretching in the
    PML parts) on the air and PML parts."""
    PTS_PER_TRI = 6
    dofs_per_node = 1

    def __init__(self, objects):
        self._setup(select(objects, "air", "pml"))

    def assemble(self):
        """H, Q (complex if a PML is present) and the density of the (last non-PML) air region."""
        has_pml = any(sd.role == "pml" for sd in self.subdomains)
        dtype = np.complex128 if has_pml else np.float64
        Hv = np.zeros(self.V_size, dtype=dtype)
        Qv = np.zeros(self.V_size, dtype=dtype)

        for sd, ec, gdofs, s0, s1, n_e in self.prep:
            He = np.empty((n_e, self.block), dtype=dtype)
            Qe = np.empty((n_e, self.block), dtype=dtype)
            N, dN, w = self.tabulation
            if sd.role == "pml":
                sr, sz, r_t = pml_coefficients(sd.layer, quadrature_points(ec, N),
                                               float(getattr(sd, "alpha", 1.0)), int(getattr(sd, "n", 2)),
                                               sd.omega_pml, sd.c)
                two_pi_rt = 2.0 * np.pi * r_t
                diffusion_diag_kernel(ec, N, dN, w, np.ascontiguousarray(
                    np.stack([two_pi_rt * (sz / sr), two_pi_rt * (sr / sz)], axis=-1)), False, He)
                mass_kernel(ec, N, dN, w, np.ascontiguousarray(two_pi_rt * sr * sz / (sd.c * sd.c)),
                             False, Qe)
            else:
                laplace_kernel(ec, N, dN, w, coefficient(1.0, ec, N).astype(dtype), True, He)
                mass_kernel(ec, N, dN, w, coefficient(1.0 / (sd.c * sd.c), ec, N).astype(dtype), True, Qe)
                rho = sd.rho

            Hv[s0:s1] = He.ravel()
            Qv[s0:s1] = Qe.ravel()

        return self._to_csr(Hv), self._to_csr(Qv), rho
