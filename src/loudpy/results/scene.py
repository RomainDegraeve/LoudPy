"""
SceneResults: the frequency response of a 3D scene (lp.FrequencyResponse on an lp.Scene).

Besides the FE pressure field "p" (one block per band mesh, read lazily from the file), it keeps
per frequency the stacked driver modal amplitudes ("q"), the coefficients of the exact DtN
expansion outside the hemisphere ("dtn") and, with cavity modes, the modal pressures
("P_cavity"); the attachments hold the driver modal bases and the cavity modes.

    res.drivers                               names of the drivers
    hp = res.driver("top")                    DriverResponse
    hp.excursion()                            Data: axial displacement of the dust-cap centre over f
    hp.displacement(at=(r, z), theta=0)       Data: (u_r, u_z, u_θ) at a meridian point
    hp.kinetic_energy(by_order=True)          Data over (f, order)
    hp.in_vacuo()                             the same driver and drive without air (DriverResponse)
    hp.surface(f=500)                         deformed 3D surface (pyvista)
    res.pressure(at), res.spl(at)             far field outside the hemisphere (exact DtN expansion),
                                              FE field inside, cavity modes in the cavity
    res.spl(at=lp.Arc(1.0, range(-180, 181, 5), center=c, axis="z", toward="x"))   directivity
    res.cavity_pressure(at)                   pressure in the cavity (cavity modes)
"""
from __future__ import annotations

import numpy as np

from loudpy.results.data import Data
from loudpy.results.results import FrequencyResults
from loudpy.results.selection import as_selection


class DriverResponse:
    """One driver of a scene: its modal amplitudes q (n_f, n_modes) over frequency and its modal basis."""

    def __init__(self, name, modes, frame, q, freqs, coil_nodes, force=None, label=""):
        self.name, self.modes, self.frame = name, modes, frame
        self.q, self.frequencies, self.coil_nodes = q, np.asarray(freqs), coil_nodes
        self.force = force                                     # modal force of the drive (or None)
        self.label = label or name

    def __repr__(self):
        return f"DriverResponse({self.label!r}, {len(self.modes)} modes, {len(self.frequencies)} frequencies)"

    def _data(self, values, coords, q, **meta):
        label = self.label if self.label != self.name else ""                # in-vacuo responses say so
        return Data(values, {"f": self.frequencies, **coords}, q, label, {"driver": self.name, **meta})

    def in_vacuo(self) -> "DriverResponse":
        """The same driver under the same drive, without air (in vacuo modal response)."""
        if self.force is None:
            raise ValueError("these results do not store the drive (re-run the study)")
        w = 2 * np.pi * self.frequencies[:, None]
        wk = 2 * np.pi * self.modes.freq[None, :]
        q0 = self.force[None, :] / (wk ** 2 + 2j * self.modes.zeta[None, :] * wk * w - w ** 2)
        return DriverResponse(self.name, self.modes, self.frame, q0, self.frequencies, self.coil_nodes, self.force,
                              f"{self.label} (in vacuo)")

    def kinetic_energy(self, by_order: bool = False) -> Data:
        """Kinetic energy ½ω²|q|² over frequency (mass-normalised modes) [J]; by_order: per azimuthal order m."""
        w2 = (2 * np.pi * self.frequencies) ** 2
        if not by_order:
            return self._data(0.5 * w2 * np.sum(np.abs(self.q) ** 2, axis=1), {}, "energy")
        orders = np.unique(self.modes.m)
        E = np.column_stack([0.5 * w2 * np.sum(np.abs(self.q[:, self.modes.m == m]) ** 2, axis=1) for m in orders])
        return self._data(E, {"order": orders}, "energy")

    def node(self, at=None) -> int:
        """Meridian node nearest `at` (r, z); default: on the axis at the top (dust-cap centre)."""
        xy = self.modes.xy
        if at is None:
            on_axis = np.where(xy[:, 0] < 1e-9)[0]
            return int(on_axis[np.argmax(xy[on_axis, 1])])
        return int(np.argmin(np.linalg.norm(xy - np.asarray(at, float), axis=1)))

    def _displacement(self, k, theta):
        U = self.modes.U[:, k, :]                                              # (n_modes, 3)
        a = self.modes.angular(theta)[:, 0]
        b = np.where(self.modes.family == "s", -np.cos(self.modes.m * theta), np.sin(self.modes.m * theta))
        b[self.modes.m == 0] = 1.0
        return np.column_stack([self.q @ (U[:, 0] * a), self.q @ (U[:, 1] * a), self.q @ (U[:, 2] * b)])

    def displacement(self, at=None, theta: float = 0.0, component: str | None = None) -> Data:
        """Displacement (u_r, u_z, u_θ) at a meridian point (r, z) of the driver, azimuth θ [rad] in its frame."""
        k = self.node(at)
        u = self._displacement(k, theta)
        where = dict(at=f"(r, z) = ({self.modes.xy[k, 0]:.4g}, {self.modes.xy[k, 1]:.4g}) m", theta=theta)
        if component is not None:
            return self._data(u[:, {"r": 0, "z": 1, "theta": 2}[component]], {}, f"u_{component}", **where)
        return self._data(u, {"component": np.array(["r", "z", "theta"])}, "u", **where)

    def excursion(self, at=None, theta: float = 0.0) -> Data:
        """Axial displacement u_z over frequency (default: dust-cap centre)."""
        return self.displacement(at, theta, "z")

    def coil_excursion(self) -> Data:
        """Mean axial displacement of the coil over frequency."""
        uz = np.mean([self._displacement(n, 0.0)[:, 1] for n in self.coil_nodes], axis=0)
        return self._data(uz, {}, "u_z", at="coil (mean)")

    def surface(self, f: float, local: bool = False, **kwargs):
        """
        Deformed 3D surface (pyvista) at the frequency nearest f, placed in the scene (or in the driver's
        own frame with local=True: axis along z). Keywords of results.surfaces.driver_surface
        (theta, n_theta, deformation, phase, u_max). Returns (surface, phase).
        """
        from loudpy.results.surfaces import driver_surface
        i = int(np.argmin(np.abs(self.frequencies - f)))
        return driver_surface(self.modes, self.q[i], (0.0, 0.0, 0.0) if local else self.frame, **kwargs)


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

    def _far(self, pts) -> np.ndarray:
        dtn = self._dtn()
        C = self._stacked("dtn")
        n_h = np.concatenate([np.asarray(b.extra["dtn_n"]) for b in self.blocks]).astype(int)
        c = self.scene["c"]
        out = np.zeros((len(self), len(pts)), complex)
        for i, f in enumerate(self.frequencies):
            out[i] = dtn.pressure(pts, C[i, :n_h[i]], 2 * np.pi * f / c)
        return out

    def driver_surfaces(self, f: float, deformation: float = 0.10, n_theta: int = 97) -> dict:
        """
        Deformed 3D surfaces (pyvista) of every driver at the frequency nearest f, at one instant and
        one scale: the phase and the displacement of the driver that moves most set them (its largest
        displacement = deformation × its diameter). {name: surface with "|u|" (re that maximum)}.
        """
        drv = [self.driver(n) for n in self.drivers]
        first = [d.surface(f, n_theta=n_theta, deformation=deformation) for d in drv]
        big = int(np.argmax([s["|u| [m]"].max() for s, _ in first]))
        ph = first[big][1]
        u_max = max(d.surface(f, n_theta=13, phase=ph)[0]["|u| [m]"].max() for d in drv)
        return {d.name: d.surface(f, n_theta=n_theta, deformation=deformation, phase=ph, u_max=u_max)[0] for d in drv}

    # ── pressure ───────────────────────────────────────────────────────────
    def _dtn(self):
        from loudpy.physics.exterior import HalfSpaceDtN
        h = self.scene.get("hemisphere")
        if not h:
            raise ValueError("these results have no exterior air (no DtN expansion)")
        return HalfSpaceDtN(h["center"], h["radius"], up=h["up"])

    def far_field(self, at) -> Data:
        """Pressure outside the hemisphere (exact DtN expansion) at points / a selection [Pa]."""
        sel = as_selection(at)
        pts = sel.points(3)
        h = self.scene.get("hemisphere")
        if not h:
            raise ValueError("these results have no exterior air (no DtN expansion)")
        r = np.linalg.norm(pts - np.asarray(h["center"]), axis=1)
        if np.any(r < h["radius"] * (1 - 1e-9)):
            raise ValueError(f"far_field needs points outside the hemisphere (radius {h['radius']:.3f} m); "
                             f"use pressure() for points inside")
        return self._pressure_data(self._far(pts), sel)

    def _pressure_data(self, P, sel):
        coords = {"f": self.frequencies}
        if sel.dim is None:
            P = P[:, 0]
        else:
            coords[sel.dim] = sel.coord
        return Data(P, coords, "p", meta=self._meta(at=sel.describe()))

    @property
    def cavity_frequencies(self) -> np.ndarray:
        """Rigid-wall frequencies of the cavity modes [Hz]."""
        cav = self.attachments.get("cavity")
        if cav is None:
            raise ValueError("these results have no cavity modes")
        return self.scene["c"] * np.sqrt(np.maximum(np.asarray(cav["kappa2"]), 0.0)) / (2 * np.pi)

    def _cavity(self, pts):
        cav = self.attachments.get("cavity")
        if cav is None:
            raise ValueError("these results have no cavity modes")
        nodes = np.asarray(cav["nodes"])
        rows = np.array([int(np.argmin(np.linalg.norm(nodes - p, axis=1))) for p in pts])
        uniq, inv = np.unique(rows, return_inverse=True)                  # increasing rows (lazy datasets)
        psi = np.asarray(cav["psi"][uniq])
        return self._stacked("P_cavity") @ psi[inv].T

    def cavity_mesh(self, cavity=None):
        """The cavity mesh (geometry.Mesh, TET10 group "air"): from the results, or from a CavityModes
        (physics.modal; for result files written before the cavity cells were stored)."""
        from loudpy.geometry.mesh import Mesh, MeshGroup
        cache = self.__dict__.setdefault("_cavity_meshes", {})
        if id(cavity) in cache:                                  # one Mesh object: plots reuse its grid
            return cache[id(cavity)][1]
        cav = self.attachments.get("cavity")
        if cavity is not None:
            nodes, tets = cavity.nodes, cavity.tets
        elif cav is not None and "tets" in cav:
            nodes, tets = np.asarray(cav["nodes"]), np.asarray(cav["tets"])
        else:
            raise ValueError("no cavity cells in these results: pass cavity=CavityModes.load(…_modes.h5)")
        mesh = Mesh(np.arange(len(nodes)), np.asarray(nodes, float), {"air": MeshGroup("air", 3, "tet10", tets)})
        cache[id(cavity)] = (cavity, mesh)
        return mesh

    def cavity_field(self, f: float, cavity=None):
        """Field: the pressure in the cavity (cavity modes) at the frequency nearest f."""
        from loudpy.results.data import Field
        cav = self.attachments["cavity"]
        i = int(np.argmin(np.abs(self.frequencies - f)))
        p = np.asarray(cav["psi"], float) @ self._stacked("P_cavity")[i]
        return Field("p", self.cavity_mesh(cavity), p, self.frequencies[i], "f", meta=self._meta(region="cavity"))

    def cavity_mode(self, n: int, cavity=None):
        """Field: the rigid-wall cavity mode n (pressure shape, coordinate = its frequency)."""
        from loudpy.results.data import Field, Quantity
        psi = np.asarray(self.attachments["cavity"]["psi"][:, n] if cavity is None else cavity.psi[:, n], float)
        return Field("p", self.cavity_mesh(cavity), psi, float(self.cavity_frequencies[n]), "f",
                     quantity=Quantity("psi", "Cavity mode shape", r"\psi"), meta={"mode": n})

    def cavity_pressure(self, at) -> Data:
        """Pressure in the cavity (cavity modes, nearest node) at points / a selection [Pa]."""
        sel = as_selection(at)
        return self._pressure_data(self._cavity(sel.points(3)), sel)

    def pressure(self, at) -> Data:
        """Pressure at points / a selection: far field outside the hemisphere, FE field inside, cavity otherwise."""
        sel = as_selection(at)
        pts = sel.points(3)
        P = np.zeros((len(self), len(pts)), complex)
        h = self.scene.get("hemisphere")
        out = np.zeros(len(pts), bool) if not h else \
            np.linalg.norm(pts - np.asarray(h["center"]), axis=1) >= h["radius"]
        if out.any():
            P[:, out] = self._far(pts[out])
        if (~out).any():
            if "p" in self.names:
                from loudpy.results.selection import Points
                P[:, ~out] = np.asarray(self.probe("p", Points(pts[~out])))
            else:
                P[:, ~out] = self._cavity(pts[~out])
        return self._pressure_data(P, sel)

    def spl(self, at, rms: bool = True) -> Data:
        """Sound pressure level [dB re 20 µPa] at points / a selection (see pressure)."""
        return self.pressure(at).spl(rms)
