"""
Physical quantities: one name, label, symbol, unit and colormap family per quantity, used by every
evaluated result (Data, Field), every axis label and every export header.

    Q = quantity("spl")         Quantity(name="spl", label="Sound pressure level", symbol="L_p", unit="dB re 20 µPa")
    Q.axis_label()              "Sound pressure level $L_p$ [dB re 20 µPa]"
    quantity("u_z")             axial displacement [m]
    quantity("u_z").part("abs") its modulus |u_z| [m]

Names of the fields and their components: "p"; "u", "u_r", "u_z", "u_theta" (and "v_…", "a_…").
Coordinates: "f", "t", "angle", "s" (distance along a line), "r", "z", "x", "y", "point", "mode".
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Quantity:
    name: str
    label: str
    symbol: str = ""
    unit: str = ""
    kind: str = "plain"                 # colormap family (loudpy.plot.style.CMAPS) / plotting defaults
    complex: bool = False               # values are complex amplitudes (harmonic)

    def axis_label(self) -> str:
        sym = f" ${self.symbol}$" if self.symbol else ""
        unit = f" [{self.unit}]" if self.unit else ""
        return f"{self.label}{sym}{unit}"

    def column(self) -> str:
        """Column name of exports: "spl [dB re 20 µPa]"."""
        return f"{self.name} [{self.unit}]" if self.unit else self.name

    def part(self, which: str) -> "Quantity":
        """The quantity of a part of a complex amplitude: "abs", "real", "imag", "phase", "db"."""
        s = self.symbol or self.name
        if which == "abs":
            return replace(self, name=f"|{self.name}|", symbol=f"|{s}|", complex=False)
        if which == "real":
            return replace(self, name=f"Re {self.name}", label=f"{self.label} (real part)",
                           symbol=rf"\mathrm{{Re}}\,{s}", complex=False)
        if which == "imag":
            return replace(self, name=f"Im {self.name}", label=f"{self.label} (imaginary part)",
                           symbol=rf"\mathrm{{Im}}\,{s}", complex=False)
        if which == "phase":
            return Quantity(f"phase {self.name}", f"Phase of {self.label.lower()}", rf"\arg\,{s}", "rad", "phase")
        if which == "phase_deg":
            return Quantity(f"phase {self.name}", f"Phase of {self.label.lower()}", rf"\arg\,{s}", "°", "phase")
        if which == "db":
            return Quantity(f"{self.name} dB", "Level", rf"20\log_{{10}}|{s}|", f"dB re 1 {self.unit}".strip(), "level")
        raise ValueError(f"unknown part {which!r}")

    def relative(self) -> "Quantity":
        """Level relative to a reference (normalised data): dB re reference, or ratio."""
        if self.unit.startswith("dB"):
            return Quantity(f"{self.name} rel", f"{self.label}, relative", rf"\Delta {self.symbol}" if self.symbol
                            else "", "dB", "level")
        return replace(self, name=f"{self.name} rel", label=f"{self.label}, normalised", unit="", complex=self.complex)


def _q(name, label, symbol="", unit="", kind="plain", complex=False):
    return Quantity(name, label, symbol, unit, kind, complex)


_COMP = {"r": ("Radial", "r"), "z": ("Axial", "z"), "theta": ("Tangential", r"\theta"),
         "x": ("x", "x"), "y": ("y", "y")}
_VEC = {"u": ("displacement", "m"), "v": ("velocity", "m/s"), "a": ("acceleration", "m/s²")}

QUANTITIES: dict[str, Quantity] = {
    # coordinates
    "f": _q("f", "Frequency", "f", "Hz"),
    "t": _q("t", "Time", "t", "s"),
    "angle": _q("angle", "Angle", r"\theta", "°"),
    "s": _q("s", "Distance along the line", "s", "m"),
    "r": _q("r", "Radius", "r", "m"),
    "z": _q("z", "Height", "z", "m"),
    "x": _q("x", "x", "x", "m"),
    "y": _q("y", "y", "y", "m"),
    "point": _q("point", "Point"),
    "mode": _q("mode", "Mode", "n"),
    "order": _q("order", "Azimuthal order", "m"),
    "component": _q("component", "Component"),
    # acoustic
    "p": _q("p", "Pressure", "p", "Pa", "pressure", True),
    "spl": _q("spl", "Sound pressure level", "L_p", "dB re 20 µPa", "spl"),
    # electrical
    "U": _q("U", "Drive voltage", "U", "V", "plain", True),
    "I_coil": _q("I_coil", "Coil current", "I", "A", "plain", True),
    "Z": _q("Z", "Electrical impedance", "Z_e", "Ω", "plain", True),
    "Ze": _q("Ze", "Blocked electrical impedance", "Z_{e,b}", "Ω", "plain", True),
    # modal / mechanical scalars
    "zeta": _q("zeta", "Damping ratio", r"\zeta", ""),
    "s_eig": _q("s_eig", "Eigenvalue", "s", "rad/s", "plain", True),
    "force": _q("force", "Force", "F", "N"),
    "q": _q("q", "Modal amplitude", "q", "", "plain", True),
    "energy": _q("energy", "Kinetic energy", "E_k", "J"),
    "share": _q("share", "Share", "", "%"),
    "level": _q("level", "Relative level", r"\Delta L", "dB", "level"),
}
# derived fields (loudpy.results.derived)
QUANTITIES.update({
    "strain": _q("strain", "Strain", r"\varepsilon", "1", "plain", True),
    "stress": _q("stress", "Stress", r"\sigma", "Pa", "plain", True),
    "von_mises": _q("von_mises", "von Mises stress", r"\sigma_\mathrm{vM}", "Pa"),
    "particle_velocity": _q("particle_velocity", "Particle velocity", "v", "m/s", "plain", True),
    "intensity": _q("intensity", "Active acoustic intensity", "I", "W/m²"),
})
for _c, _lab in {"rr": "radial", "zz": "axial", "rz": "shear", "tt": "hoop"}.items():
    QUANTITIES[f"strain_{_c}"] = _q(f"strain_{_c}", f"Strain, {_lab}", rf"\varepsilon_{{{_c}}}", "1", "plain", True)
    QUANTITIES[f"stress_{_c}"] = _q(f"stress_{_c}", f"Stress, {_lab}", rf"\sigma_{{{_c}}}", "Pa", "plain", True)
for _c in ("r", "z"):
    QUANTITIES[f"particle_velocity_{_c}"] = _q(f"particle_velocity_{_c}", f"Particle velocity, {_c}", f"v_{_c}",
                                               "m/s", "plain", True)
    QUANTITIES[f"intensity_{_c}"] = _q(f"intensity_{_c}", f"Active intensity, {_c}", f"I_{_c}", "W/m²")
for _f, (_what, _unit) in _VEC.items():
    QUANTITIES[_f] = _q(_f, _what.capitalize(), _f, _unit, "displacement", True)
    for _c, (_adj, _sym) in _COMP.items():
        QUANTITIES[f"{_f}_{_c}"] = _q(f"{_f}_{_c}", f"{_adj} {_what}", f"{_f}_{{{_sym}}}", _unit, "displacement", True)


def quantity(q) -> Quantity:
    """The Quantity of a name (a Quantity is returned as is; an unknown name gets a plain quantity)."""
    if isinstance(q, Quantity):
        return q
    if q in QUANTITIES:
        return QUANTITIES[q]
    return Quantity(str(q), str(q))


COMPONENTS = {"r": 0, "z": 1, "theta": 2}           # columns of the axisymmetric vector fields (u, v, a)
