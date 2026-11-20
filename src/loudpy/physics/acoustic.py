"""
Acoustic operators of the air regions (roles "air" and "pml"): H and Q, 1 DOF per node.

A PML region (role "pml") carries its layer geometry in `part.layer` (geometry.naming.PMLLayer)
and its absorption parameters (alpha, n, omega_pml); its H and Q are complex.
"""
from __future__ import annotations

import numpy as np

from loudpy.fem.assembly import ElementAssembler
from loudpy.fem.batch import _batch_H, _batch_Q, _batch_H_pml, _batch_Q_pml
from loudpy.physics.registry import select


class AcouAssembler(ElementAssembler):
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
            sign_val = 1.0                      # same formulation in every air region

            if sd.role == "pml":
                L = sd.layer
                sigma_max = float(getattr(sd, "alpha", 1.0))
                p_order = int(getattr(sd, "n", 2))
                _batch_H_pml(ec, L.use_r, L.r0, L.tr, L.sgn_r, L.use_z, L.z0, L.tz, L.sgn_z,
                             sigma_max, p_order, sd.omega_pml, sd.c, sign_val, He)
                _batch_Q_pml(ec, sd.c, L.use_r, L.r0, L.tr, L.sgn_r, L.use_z, L.z0, L.tz, L.sgn_z,
                             sigma_max, p_order, sd.omega_pml, sign_val, Qe)
            else:
                _batch_H(ec, sign_val, He)
                _batch_Q(ec, sd.c, sign_val, Qe)
                rho = sd.rho

            Hv[s0:s1] = He.ravel()
            Qv[s0:s1] = Qe.ravel()

        return self._to_csr(Hv), self._to_csr(Qv), rho
