"""
SceneResults: the frequency response of a 3D scene (lp.FrequencyResponse on an lp.Scene).

Besides the FE pressure field "p" (one block per band mesh, read lazily from the file), it keeps
per frequency the stacked driver modal amplitudes ("q"), the coefficients of the exact DtN
expansion outside the hemisphere ("dtn") and, with cavity modes, the modal pressures
("P_cavity"); the attachments hold the driver modal bases and the cavity modes.

    res.drivers                          names of the drivers
    res.driver("top").excursion()        axial displacement of the dust-cap centre over frequency
    res.driver("top").surface(f=500)     deformed 3D surface of the driver (pyvista)
    res.driver("top").in_vacuo()         modal amplitudes of the same drive without air
    res.driver("top").kinetic_energy(by_order=True)
    res.far_field(points)                pressure outside the hemisphere, from the DtN expansion
    res.spl(at=point)                    SPL: far field outside the hemisphere, FE field inside
    res.cavity_pressure(point)           pressure in the cavity (cavity modes)
"""
from __future__ import annotations

import numpy as np

from loudpy.results.results import FrequencyResults, spl as _spl


class DriverResponse:
    """One driver of a scene: its modal amplitudes over frequency and its modal basis."""

    def __init__(self, name, modes, frame, q, freqs, coil_nodes, force=None):
        self.name, self.modes, self.frame = name, modes, frame
        self.q, self.frequencies, self.coil_nodes = q, freqs, coil_nodes
        self.force = force                                     # modal force of the drive (or None)

    def in_vacuo(self) -> np.ndarray:
        """Modal amplitudes (n_f, n_modes) of the same driver under the same drive, without air."""
        if self.force is None:
            raise ValueError("these results do not store the drive (re-run the study)")
        w = 2 * np.pi * np.asarray(self.frequencies)[:, None]
        wk = 2 * np.pi * self.modes.freq[None, :]
        return self.force[None, :] / (wk ** 2 + 2j * self.modes.zeta[None, :] * wk * w - w ** 2)

    def kinetic_energy(self, q=None, by_order: bool = False):
        """Kinetic energy ½ω²|q|² over frequency (mass-normalised modes); by_order: {m: E_m}."""
        q = self.q if q is None else q
        w2 = (2 * np.pi * np.asarray(self.frequencies)) ** 2
        if not by_order:
            return 0.5 * w2 * np.sum(np.abs(q) ** 2, axis=1)
        return {int(m): 0.5 * w2 * np.sum(np.abs(q[:, self.modes.m == m]) ** 2, axis=1)
                for m in np.unique(self.modes.m)}

    def node(self, at=None) -> int:
        """Meridian node nearest `at` (r, z); default: on the axis at the top (dust-cap centre)."""
        xy = self.modes.xy
        if at is None:
            on_axis = np.where(xy[:, 0] < 1e-9)[0]
            return int(on_axis[np.argmax(xy[on_axis, 1])])
        return int(np.argmin(np.linalg.norm(xy - np.asarray(at, float), axis=1)))

    def displacement(self, at=None, theta: float = 0.0) -> np.ndarray:
        """(n_f, 3) displacement (u_r, u_z, u_θ) at a meridian point, azimuth θ (driver frame)."""
        k = self.node(at)
        U = self.modes.U[:, k, :]                                              # (n_modes, 3)
        a = self.modes.angular(theta)[:, 0]
        b = np.where(self.modes.family == "s", -np.cos(self.modes.m * theta), np.sin(self.modes.m * theta))
        b[self.modes.m == 0] = 1.0
        return np.column_stack([self.q @ (U[:, 0] * a), self.q @ (U[:, 1] * a), self.q @ (U[:, 2] * b)])

    def excursion(self, at=None) -> np.ndarray:
        """Axial displacement u_z over frequency (default: dust-cap centre)."""
        return self.displacement(at)[:, 1]

    def coil_excursion(self) -> np.ndarray:
        """Mean axial displacement of the coil over frequency."""
        uz = np.mean([self.displacement(self.modes.xy[n])[:, 1] for n in self.coil_nodes], axis=0)
        return uz

    def surface(self, f: float, **kwargs):
        """Deformed 3D surface at the frequency nearest f (driver_surface keyword arguments)."""
        from loudpy.results.surfaces import driver_surface
        i = int(np.argmin(np.abs(self.frequencies - f)))
        return driver_surface(self.modes, self.q[i], self.frame, **kwargs)


class SceneResults(FrequencyResults):
    @property
    def scene(self) -> dict:
        return self.attachments.get("scene", {}).get("meta", {})

    @property
    def drivers(self) -> list[str]:
        return [d["name"] for d in self.scene.get("drivers", [])]

    def _stacked(self, name):
        rows = [np.asarray(b.extra[name]) for b in self.blocks if name in b.extra]
        if not rows:
            raise KeyError(f"no {name!r} in these results")
        n = max(r.shape[1] for r in rows)
        out = np.zeros((len(self), n), rows[0].dtype)
        i = 0
        for r in rows:
            out[i:i + len(r), :r.shape[1]] = r
            i += len(r)
        return out

    @property
    def q(self) -> np.ndarray:
        """Stacked driver modal amplitudes (n_f, n_modes of all drivers [+ extra coordinates])."""
        return self._stacked("q")

    def driver(self, name: str) -> DriverResponse:
        from loudpy.geometry.frame import Frame
        from loudpy.physics.modal import DriverModes
        d = next((d for d in self.scene["drivers"] if d["name"] == name), None)
        if d is None:
            raise KeyError(f"no driver {name!r} (drivers: {self.drivers})")
        a = self.attachments[d["modes"]]
        modes = DriverModes(a["xy"], a["tris"], a["m"], np.asarray(a["family"]).astype(str), a["freq"], a["zeta"],
                            np.asarray(a["U"]))
        frame = Frame(d["origin"], np.asarray(d["rotation"]))
        sl = slice(d["offset"], d["offset"] + len(modes))
        q = self.q[:, sl]
        F = self.attachments.get("drive", {}).get("F")
        return DriverResponse(name, modes, frame, q, self.frequencies, np.asarray(a["coil_nodes"]),
                              None if F is None else np.asarray(F)[sl])

    # ── pressure ───────────────────────────────────────────────────────────
    def _dtn(self):
        from loudpy.physics.exterior import HalfSpaceDtN
        h = self.scene.get("hemisphere")
        if not h:
            raise ValueError("these results have no exterior air (no DtN expansion)")
        return HalfSpaceDtN(h["center"], h["radius"], up=h["up"])

    def far_field(self, points) -> np.ndarray:
        """Pressure (n_f, n_points) at points outside the hemisphere, from the exact DtN expansion."""
        dtn = self._dtn()
        pts = np.atleast_2d(np.asarray(points, float))
        r = np.linalg.norm(pts - np.asarray(dtn.center), axis=1)
        if np.any(r < dtn.R * (1 - 1e-9)):
            raise ValueError(f"far_field needs points outside the hemisphere (radius {dtn.R:.3f} m); "
                             f"use probe() inside")
        C = self._stacked("dtn")
        n_h = np.concatenate([np.asarray(b.extra["dtn_n"]) for b in self.blocks]).astype(int)
        c = self.scene["c"]
        out = np.zeros((len(self), len(pts)), complex)
        for i, f in enumerate(self.frequencies):
            out[i] = dtn.pressure(pts, C[i, :n_h[i]], 2 * np.pi * f / c)
        return out

    @property
    def cavity_frequencies(self) -> np.ndarray:
        """Rigid-wall frequencies of the cavity modes [Hz]."""
        cav = self.attachments.get("cavity")
        if cav is None:
            raise ValueError("these results have no cavity modes")
        return self.scene["c"] * np.sqrt(np.maximum(np.asarray(cav["kappa2"]), 0.0)) / (2 * np.pi)

    def cavity_pressure(self, point) -> np.ndarray:
        """Pressure over frequency at the cavity node nearest `point` (cavity modes)."""
        cav = self.attachments.get("cavity")
        if cav is None:
            raise ValueError("these results have no cavity modes")
        k = int(np.argmin(np.linalg.norm(np.asarray(cav["nodes"]) - np.asarray(point, float), axis=1)))
        return self._stacked("P_cavity") @ np.asarray(cav["psi"][k])

    def spl(self, at, rms: bool = True) -> np.ndarray:
        """SPL at `at`: far field outside the hemisphere, nearest FE node inside (or the cavity)."""
        at = np.asarray(at, float)
        h = self.scene.get("hemisphere")
        if h and np.linalg.norm(at - np.asarray(h["center"])) >= h["radius"]:
            p = self.far_field(at)[:, 0]
        elif "p" in self.names:
            p = self.probe(at, "p")
        else:
            p = self.cavity_pressure(at)
        return _spl(p, rms)
