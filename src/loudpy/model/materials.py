"""
Materials: values in SI units, with their reference temperature and their source, loaded once from a
JSON library and checked (loudpy.checks).

    lp.materials["Paper"]                       the default library (data/materials.json)
    lp.load_materials("my_materials.json")      another library
    lp.Material("Steel", rho=7850, E=2.1e11, nu=0.3)      a material defined in the script
    lp.materials["Paper"].with_(E=1.2e9)        a variant
    lp.air(temperature=298.15, humidity=0.5)    air at other conditions (ideal-gas mixture model)

A solid has rho, E, nu; a fluid has rho, c. Damping is not a material constant: a material may carry
*default damping fits* (``rayleigh=(alpha, beta)``, ``loss_factor=eta``) that lp.Rayleigh() /
lp.Hysteretic() use when no coefficient is given. PML settings belong to lp.PML, not to the air.

Air: the library's "Air" is computed from its conditions (20 °C, 101.325 kPa, dry) by :func:`air`;
2D models, 3D scenes and the low-level toolkit all use it (RHO_AIR, C_AIR are its values).

Library file (format 2)::

    {"format": "loudpy-materials/2",
     "materials": {
       "Paper": {"kind": "solid", "rho": 1200, "E": 1.0e9, "nu": 0.30,
                 "damping": {"rayleigh": [18.0, 4.6e-6], "loss_factor": 0.03},
                 "temperature": 293.15, "source": "typical value, unverified"},
       "Air":   {"kind": "fluid", "model": "air", "temperature": 293.15, "pressure": 101325.0,
                 "humidity": 0.0}}}

Libraries of format 1 (flat entries with "type", "eta", "alpha_ray", "beta_ray") are still read.
"""
from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path

from loudpy.errors import LoudpyWarning, MaterialError, SelectionError

DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / "data" / "materials.json"
FORMAT = "loudpy-materials/2"
UNVERIFIED = "typical value, unverified"

# ── air: ideal-gas mixture of dry air and water vapour ───────────────────────
R_GAS = 8.314462618               # molar gas constant [J/(mol·K)]
M_DRY_AIR = 0.0289647             # molar mass of dry air [kg/mol]
M_WATER = 0.01801528              # molar mass of water [kg/mol]
CP_DRY_AIR = 3.5 * R_GAS          # molar heat capacity at constant pressure, dry air (γ = 1.4) [J/(mol·K)]
CP_WATER = 4.0 * R_GAS            # the same for water vapour (γ = 4/3)
T_REF, P_REF = 293.15, 101325.0   # standard conditions of loudpy: 20 °C, 1 atm


def saturation_vapour_pressure(temperature: float) -> float:
    """Saturation vapour pressure of water over a liquid surface [Pa] (Buck 1996), temperature in K."""
    t = temperature - 273.15
    return 611.21 * math.exp((18.678 - t / 234.5) * (t / (257.14 + t)))


@dataclass(frozen=True)
class Material:
    """
    A material in SI units.

    name               identifier
    kind               "solid" or "fluid"
    rho                density [kg/m³]
    E, nu              Young's modulus [Pa], Poisson's ratio (solid)
    c                  speed of sound [m/s] (fluid)
    rayleigh           default Rayleigh damping fit (alpha [1/s], beta [s]): C = alpha M + beta K (solid)
    loss_factor        default hysteretic loss factor eta (solid)
    temperature        reference temperature of the values [K]
    pressure, humidity static pressure [Pa] and relative humidity (0–1) of a gas computed by air()
    source             where the values come from
    """
    name: str
    kind: str = "solid"
    rho: float | None = None
    E: float | None = None
    nu: float | None = None
    c: float | None = None
    rayleigh: tuple[float, float] | None = None
    loss_factor: float | None = None
    temperature: float | None = None
    pressure: float | None = None
    humidity: float | None = None
    source: str = ""

    def __post_init__(self):
        from loudpy.checks import check
        if self.kind not in ("solid", "fluid"):
            raise MaterialError(f"material {self.name!r}: kind must be 'solid' or 'fluid', got {self.kind!r}")
        if self.rayleigh is not None:
            if len(self.rayleigh) != 2:
                raise MaterialError(f"material {self.name!r}: rayleigh must be (alpha, beta)")
            object.__setattr__(self, "rayleigh", (float(self.rayleigh[0]), float(self.rayleigh[1])))
            check(self.rayleigh[0], "rayleigh_alpha", f"material {self.name!r}: rayleigh alpha")
            check(self.rayleigh[1], "rayleigh_beta", f"material {self.name!r}: rayleigh beta")
        for attr, kind in (("rho", "density"), ("E", "young_modulus"), ("nu", "poisson_ratio"),
                           ("c", "speed_of_sound"), ("loss_factor", "loss_factor"), ("temperature", "temperature"),
                           ("pressure", "pressure"), ("humidity", "relative_humidity")):
            check(getattr(self, attr), kind, f"material {self.name!r}: {attr}")

    def with_(self, **overrides) -> "Material":
        """A copy with some properties changed (None values are ignored)."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def require(self, *names) -> dict:
        """{name: value} of properties that must be set; a clear error otherwise."""
        missing = [n for n in names if getattr(self, n) is None]
        if missing:
            raise MaterialError(f"material {self.name!r} has no {', '.join(missing)}: give "
                                f"{', '.join(m + '=' for m in missing)} or add it to the material library")
        return {n: getattr(self, n) for n in names}

    # ── library entries ────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict) -> "Material":
        """A material from a library entry (format 2, or the flat format 1)."""
        d = dict(d)
        if d.get("model") == "air":
            return air(d.get("temperature", T_REF), d.get("pressure", P_REF), d.get("humidity", 0.0),
                       name=d.get("name", "Air"))
        if "type" in d or any(k in d for k in ("eta", "alpha_ray", "beta_ray")):
            d = _from_format1(d)
        damping = d.pop("damping", {}) or {}
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise MaterialError(f"material {d.get('name')!r}: unknown properties {sorted(unknown)}")
        if "rayleigh" in damping:
            d["rayleigh"] = tuple(damping["rayleigh"])
        if "loss_factor" in damping:
            d["loss_factor"] = damping["loss_factor"]
        d.setdefault("source", UNVERIFIED)
        return cls(**d)

    def to_dict(self) -> dict:
        """The library entry (format 2) of this material."""
        if self.kind == "fluid" and self.pressure is not None:
            return {"kind": "fluid", "model": "air", "temperature": self.temperature, "pressure": self.pressure,
                    "humidity": self.humidity}
        out = {k: getattr(self, k) for k in ("kind", "rho", "E", "nu", "c") if getattr(self, k) is not None}
        damping: dict = {}
        if self.rayleigh is not None:
            damping["rayleigh"] = list(self.rayleigh)
        if self.loss_factor is not None:
            damping["loss_factor"] = self.loss_factor
        if damping:
            out["damping"] = damping
        if self.temperature is not None:
            out["temperature"] = self.temperature
        out["source"] = self.source
        return out


def _from_format1(d: dict) -> dict:
    """Format-1 entry (flat: type, eta, alpha_ray, beta_ray, PML alpha / n) → format-2 fields."""
    out = {"name": d.get("name"), "kind": {"elastic": "solid", "acoustic": "fluid"}.get(str(d.get("type", d.get("kind"))),
                                                                                         d.get("kind", "solid"))}
    for k in ("rho", "E", "nu", "c", "temperature", "source"):
        if k in d:
            out[k] = d[k]
    damping = {}
    if "alpha_ray" in d and "beta_ray" in d:
        damping["rayleigh"] = [d["alpha_ray"], d["beta_ray"]]
    if "eta" in d:
        damping["loss_factor"] = d["eta"]
    if damping:
        out["damping"] = damping
    if "alpha" in d or "n" in d:
        warnings.warn(f"material {d.get('name')!r}: the PML settings 'alpha' / 'n' of format-1 libraries are ignored; "
                      f"give lp.PML(strength=…, order=…)", LoudpyWarning, stacklevel=4)
    return out


def air(temperature: float = T_REF, pressure: float = P_REF, humidity: float = 0.0, name: str = "Air") -> Material:
    """
    Air at a temperature [K], static pressure [Pa] and relative humidity (0–1), as an ideal-gas mixture
    of dry air and water vapour:

        x_w = humidity · p_sat(T) / p                  mole fraction of water vapour (Buck 1996 for p_sat)
        M   = (1 − x_w) M_air + x_w M_water            molar mass
        γ   = c_p / (c_p − R),  c_p = (1 − x_w) 3.5 R + x_w 4 R
        ρ   = p M / (R T),      c = √(γ R T / M)

    20 °C, 101.325 kPa, dry: ρ = 1.2041 kg/m³, c = 343.24 m/s. The speed of sound is within 0.05 % of
    Cramer's correlation (JASA 93, 1993) over 0–40 °C, 80–101 kPa, 0–100 % RH (tests/test_verification.py).
    """
    from loudpy.checks import check
    check(temperature, "temperature", "air temperature")
    check(pressure, "pressure", "air pressure")
    check(humidity, "relative_humidity", "air humidity")
    x_w = humidity * saturation_vapour_pressure(temperature) / pressure
    molar = (1 - x_w) * M_DRY_AIR + x_w * M_WATER
    cp = (1 - x_w) * CP_DRY_AIR + x_w * CP_WATER
    gamma = cp / (cp - R_GAS)
    return Material(name, kind="fluid", rho=pressure * molar / (R_GAS * temperature),
                    c=math.sqrt(gamma * R_GAS * temperature / molar), temperature=float(temperature),
                    pressure=float(pressure), humidity=float(humidity),
                    source="ideal-gas mixture of dry air and water vapour (loudpy.model.materials.air)")


STANDARD_AIR = air()
RHO_AIR, C_AIR = STANDARD_AIR.rho, STANDARD_AIR.c      # air at 20 °C, 101.325 kPa, dry (toolkit defaults)


# ── libraries ────────────────────────────────────────────────────────────────
class MaterialLibrary(Mapping):
    """Read-only mapping name → Material, loaded from a JSON library (format 2, or format 1)."""

    def __init__(self, materials: dict[str, Material], path: Path | None = None):
        self._m = dict(materials)
        self.path = Path(path) if path is not None else None

    @classmethod
    def load(cls, path) -> "MaterialLibrary":
        """Read a material library file (JSON, formats 1 and 2)."""
        path = Path(path)
        with open(path) as fh:
            data = json.load(fh)
        entries = data["materials"] if isinstance(data.get("materials"), dict) else data
        return cls({k: Material.from_dict({"name": k, **v}) for k, v in entries.items()}, path)

    def save(self, path) -> Path:
        """Write the library (format 2)."""
        path = Path(path)
        data = {"format": FORMAT, "materials": {k: m.to_dict() for k, m in self._m.items()}}
        path.write_text(json.dumps(data, indent=2) + "\n")
        return path

    def __getitem__(self, name) -> Material:
        try:
            return self._m[name]
        except KeyError:
            raise SelectionError(f"no material {name!r} in {self.path or 'the library'} "
                                 f"(available: {', '.join(sorted(self._m))})") from None

    def __iter__(self):
        return iter(self._m)

    def __len__(self):
        return len(self._m)

    def with_(self, *materials: Material, path=None) -> "MaterialLibrary":
        """A new library with these materials added or replaced (by name); path: where save() writes it by
        default (the library itself is read-only).
            mine = lp.materials.with_(dataclasses.replace(lp.materials["Paper"], E=4.2e9), path="my.json")"""
        lib = dict(self._m)
        for m in materials:
            if not isinstance(m, Material):
                raise MaterialError(f"MaterialLibrary.with_: lp.Material objects, got {type(m).__name__}")
            lib[m.name] = m
        return MaterialLibrary(lib, path if path is not None else self.path)

    def resolve(self, material: "str | Material") -> Material:
        """The Material itself, or the library's material of that name."""
        return material if isinstance(material, Material) else self[material]

    def __repr__(self):
        return f"MaterialLibrary({len(self)} materials from {self.path})"


def load_materials(path=None) -> MaterialLibrary:
    """The material library at `path` (default: loudpy's data/materials.json)."""
    return MaterialLibrary.load(path or DEFAULT_LIBRARY)


class _DefaultLibrary(Mapping):
    """lp.materials: the default library, read on first use."""
    _lib = None

    def _get(self):
        if self._lib is None:
            type(self)._lib = load_materials()
        return self._lib

    def __getitem__(self, k):
        return self._get()[k]

    def __iter__(self):
        return iter(self._get())

    def __len__(self):
        return len(self._get())

    def __repr__(self):
        return repr(self._get())


materials = _DefaultLibrary()
