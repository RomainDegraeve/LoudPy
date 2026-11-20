"""
Materials: values with SI units, loaded once from a JSON library and checked.

    lp.materials["Paper"]                       the default library (data/materials.json)
    lp.load_materials("my_materials.json")      another library
    lp.Material("Steel", rho=7850, E=2.1e11, nu=0.3)      a material defined in the script
    lp.materials["Paper"].with_(E=1.2e9)        a variant
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path

# Air of the 3D scenes (lp.Scene defaults) [kg/m³], [m/s]
RHO_AIR, C_AIR = 1.2, 343.0

DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / "data" / "materials.json"


@dataclass(frozen=True)
class Material:
    """
    name        : identifier
    kind        : "elastic" or "acoustic"
    rho         : density [kg/m³]
    E, nu       : Young's modulus [Pa], Poisson's ratio (elastic)
    eta         : loss factor (elastic, hysteretic damping)
    alpha_ray, beta_ray : Rayleigh damping C = alpha_ray·M + beta_ray·K (elastic)
    c           : speed of sound [m/s] (acoustic)
    alpha, n    : default PML strength and profile order (acoustic)
    """
    name: str
    kind: str = "elastic"
    rho: float | None = None
    E: float | None = None
    nu: float | None = None
    eta: float | None = None
    alpha_ray: float | None = None
    beta_ray: float | None = None
    c: float | None = None
    alpha: float | None = None
    n: float | None = None

    def with_(self, **overrides) -> "Material":
        """A copy with some properties changed (None values are ignored)."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def require(self, *names) -> dict:
        """{name: value} of properties that must be set; a clear error otherwise."""
        missing = [n for n in names if getattr(self, n) is None]
        if missing:
            raise ValueError(f"material {self.name!r} has no {', '.join(missing)}: give "
                             f"{', '.join(m + '=' for m in missing)} or add it to the material library")
        return {n: getattr(self, n) for n in names}

    @classmethod
    def from_dict(cls, d: dict) -> "Material":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known - {"type"}
        if unknown:
            raise ValueError(f"material {d.get('name')!r}: unknown properties {sorted(unknown)}")
        kw = {k: v for k, v in d.items() if k in known}
        if "type" in d and "kind" not in d:
            kw["kind"] = d["type"]
        return cls(**kw)


class MaterialLibrary(Mapping):
    """Read-only mapping name → Material, loaded from a JSON file {name: {properties}}."""

    def __init__(self, materials: dict[str, Material], path: Path | None = None):
        self._m = dict(materials)
        self.path = path

    @classmethod
    def load(cls, path) -> "MaterialLibrary":
        path = Path(path)
        with open(path) as fh:
            data = json.load(fh)
        return cls({k: Material.from_dict({"name": k, **v}) for k, v in data.items()}, path)

    def __getitem__(self, name) -> Material:
        try:
            return self._m[name]
        except KeyError:
            raise KeyError(f"no material {name!r} in {self.path or 'the library'} "
                           f"(available: {', '.join(sorted(self._m))})") from None

    def __iter__(self):
        return iter(self._m)

    def __len__(self):
        return len(self._m)

    def resolve(self, material: "str | Material") -> Material:
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
