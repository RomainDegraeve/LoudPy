"""
What a simulation is made of, described for programs (the interface, checks, documentation): every building block a
script composes — model, regions, boundaries, mesh controls, materials, damping, drives, selectors, studies and their
options, transient signals, a 3D scene's cabinet, drivers and air — with its parameters: kind of value, unit, help,
choices, limits, default and the form group it is shown in. The defaults are read from the code (the signatures), and
the tests check that every parameter of every block is described and every public building block is in the catalog:
the description cannot drift from the API without a test failing.

    lp.schema.blocks("regions")              [Block Solid, Block Air, Block PML]
    b = lp.schema.block("PML")               b.params: (Param name, Param on, …, Param thickness [m], …)
    node = lp.schema.defaults("Solid")       {"type": "Solid", "params": {"name": "solid", "material": …}}
    node = lp.schema.describe(lp.Solid("cone", "Paper", mesh_size=1e-3))
                                             {"type": "Solid", "params": {"name": "cone", "material": "Paper",
                                              "mesh_size": 0.001}}       (what differs from the defaults)
    lp.schema.check(node)                    [] or the problems ("Solid.mesh_size: > 0, got -1")
    obj = lp.schema.build(node)              lp.Solid("cone", "Paper", mesh_size=0.001) again
    study, solve_kwargs = lp.schema.build(fr_node, model=model)       a study: its object and what .solve takes

A node is plain data (JSON): {"type": <block type>, "params": {name: value}}. Values: numbers, strings, booleans, None,
lists, nested nodes (a damping, a drive, a selector, a profile …), {"P": "h", "scale": …, "offset": …} for a parameter
reference lp.P, {"complex": [re, im]} for a complex amplitude. Block types are written as in a script after `lp.`:
"Solid", "cabinets.Box", "Scene.cavity", "Driver.from_modes", "signals.sine".

Sections (where a block goes): model, materials, regions, boundaries, mesh_controls, damping, drives,
selectors, profiles, studies, study_options, signals, scene, scene.cabinet, scene.cabinet_parts, scene.drivers,
scene.placement, scene.air, scene.walls, scene.panels.
"""
from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from loudpy.errors import ModelError


class SchemaError(ModelError):
    """A node that does not describe a building block (unknown type or parameter, a value of the wrong kind)."""


REQUIRED = inspect.Parameter.empty                     # the default of a parameter without one

KINDS = {
    "text": "a string", "number": "a real number", "integer": "an integer", "bool": "true / false",
    "choice": "one of the choices", "complex": "a real or complex number ({'complex': [re, im]})",
    "file": "a file path", "point": "coordinates (a list of numbers)", "pair": "two numbers",
    "range": "a number or two numbers (min, max)", "numbers": "a list of numbers", "integers": "a list of integers",
    "texts": "a list of strings", "pairs": "a list of number pairs", "mapping": "a mapping name → value",
    "selection": "a group name or a selector node", "material": "a library name or a Material node",
    "block": "a node of the section `of`", "blocks": "a list of nodes of the section `of` (or points)",
    "frequencies": "a list of frequencies [Hz], or {'from', 'to', 'points', 'spacing': 'log' | 'linear'}",
    "signal": "a list of samples or a signals node",
}
_TUPLE_KINDS = {"point", "pair", "range"}


@dataclass(frozen=True)
class Param:
    """One parameter of a block. kind: KINDS; unit: SI unit of the value; group: the form section it is shown in;
    optional: None is allowed (an override: "the material's"); when: the values of the block's other parameters
    it applies with (Param.applies: a form shows it only then); min / max: values must be > min (≥ with
    min_inclusive) and < max (≤ with max_inclusive) — the block refuses the others too (a test checks it);
    of: for kind "block" / "blocks", the section of the nested nodes; stage: "solve" for an argument of a study's
    .solve(); key: the block's key parameter (an item type's, shown next to its type in a table: a solid's
    material); default: from the code (REQUIRED: none)."""
    name: str
    kind: str
    unit: str = ""
    help: str = ""
    group: str = ""
    choices: tuple = ()
    min: float | None = None
    min_inclusive: bool = False
    max: float | None = None
    max_inclusive: bool = False
    optional: bool = False
    of: str = ""
    stage: str = ""
    key: bool = False
    default: Any = REQUIRED
    when: tuple = ()                    # when it applies: alternatives, each ((other param, allowed values), …)

    def applies(self, values: dict) -> bool:
        """Whether the parameter applies given the block's other values (`when`: any alternative whose every
        (parameter, allowed values) holds; none: always). A form hides it otherwise; its value is kept."""
        if not self.when:
            return True
        return any(all(values.get(k) in allowed for k, allowed in alt) for alt in self.when)

    @property
    def required(self) -> bool:
        """True when the code gives it no default (a value must be given)."""
        return self.default is REQUIRED

    def limits(self) -> str:
        """The allowed range as text: "> 0", "≥ 0, ≤ 1", "> -1, < 0.5" ("" without limits)."""
        out = []
        if self.min is not None:
            out.append(f"{'≥' if self.min_inclusive else '>'} {self.min:g}")
        if self.max is not None:
            out.append(f"{'≤' if self.max_inclusive else '<'} {self.max:g}")
        return ", ".join(out)

    def label(self) -> str:
        """"thickness [m]": the name and the unit."""
        return self.name.replace("_", " ") + (f" [{self.unit}]" if self.unit else "")


@dataclass(frozen=True)
class Block:
    """A building block: its type (as written after `lp.`), section, summary, parameters and how it is made.
    context: arguments the surroundings give (the model a study runs on, the scene a setting applies to);
    code_only: arguments a script can give that the description leaves out, with why; facts: what an item type is
    (loudpy.model.facts: what it selects, its medium, role, where it fits), None for other blocks."""
    type: str
    section: str
    summary: str
    params: tuple[Param, ...]
    target: Callable = field(repr=False, compare=False)
    context: tuple[str, ...] = ()
    code_only: dict = field(default_factory=dict)
    solve: Callable | None = field(default=None, repr=False, compare=False)
    varargs: str = ""                                  # a parameter passed as *args (Group(*labels))
    facts: Any = field(default=None, repr=False, compare=False)

    @property
    def key(self) -> Param | None:
        """Its key parameter (an item type's: the one a table shows next to its type), or None."""
        return next((p for p in self.params if p.key), None)

    def param(self, name: str) -> Param:
        """The parameter called `name` (SchemaError if there is none)."""
        for p in self.params:
            if p.name == name:
                return p
        raise SchemaError(f"{self.type} has no parameter {name!r}: {[p.name for p in self.params]}")

    def groups(self) -> list[str]:
        """The form groups of its parameters, in order."""
        out: list[str] = []
        for p in self.params:
            if p.group and p.group not in out:
                out.append(p.group)
        return out


def _p(name, kind, unit="", help="", group="", **kw) -> Param:
    return Param(name, kind, unit, help, group, **kw)


def _signature(fn) -> dict:
    """{name: inspect.Parameter} of a callable (a dataclass: its fields, default factories called)."""
    sig = dict(inspect.signature(fn).parameters)
    sig.pop("self", None)
    if inspect.isclass(fn) and dataclasses.is_dataclass(fn):
        for f in dataclasses.fields(fn):
            if f.name in sig and f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                sig[f.name] = sig[f.name].replace(default=f.default_factory())  # type: ignore[misc]
    return sig


def _make(type_, section, target, params, *, context=(), code_only=None, solve=None, signature=None,
          varargs="") -> Block:
    sig = _signature(signature or target)
    solve_sig = _signature(solve) if solve is not None else {}
    filled = []
    for p in params:
        src = solve_sig if p.stage == "solve" else sig
        par = src.get(p.name)
        if par is None:
            raise SchemaError(f"{type_}.{p.name} is described but not in the code's signature")
        d = par.default
        filled.append(replace(p, default=REQUIRED if d is inspect.Parameter.empty or par.kind is
                              inspect.Parameter.VAR_POSITIONAL else d))
    summary = (inspect.getdoc(target) or "").strip().split("\n\n")[0].replace("\n", " ")
    from loudpy.model.facts import Facts
    facts = getattr(target, "facts", None) if inspect.isclass(target) else None
    facts = facts if isinstance(facts, Facts) else None
    if facts is not None and facts.key is not None:
        if facts.key not in {p.name for p in filled}:
            raise SchemaError(f"{type_}: its key parameter {facts.key!r} is not described")
        filled = [replace(p, key=p.name == facts.key) for p in filled]
    return Block(type_, section, summary, tuple(filled), target, tuple(context), dict(code_only or {}), solve,
                 varargs, facts)


# ── the catalog ────────────────────────────────────────────────────────────────────────────────────────────────────
def _circuit_ref(circuits, name):
    """A study's drive naming one of the simulation's circuits (built from the document's "circuits")."""
    if name not in (circuits or {}):
        raise SchemaError(f"no circuit {name!r} in the simulation (circuits: {sorted(circuits or {})})")
    return circuits[name]


def _catalog() -> dict[str, Block]:
    import loudpy as lp
    from loudpy.geometry import selectors as sel
    from loudpy.scene import cabinets as cab
    from loudpy import circuit as circ

    NAME = _p("name", "text", help="a label: nothing physical is read from it", group="Identity")
    ON = _p("on", "selection", help="the groups (or a selector) it applies to; default: the groups named like it",
            group="Selection", optional=True)
    ON_CURVES = replace(ON, help="the curves (groups or a selector) it applies to; default: the groups named like it")
    MATERIAL = _p("material", "material", help="a name of the material library, or a Material", group="Material")
    MESH = _p("mesh_size", "number", "m", "largest element size in these domains", "Mesh", min=0)
    RHO = _p("rho", "number", "kg/m³", "density; empty: the material's", "Overrides", optional=True, min=0)
    C = _p("c", "number", "m/s", "speed of sound; empty: the material's", "Overrides", optional=True, min=0)
    OUTPUT = _p("output", "block", help="what the study keeps: fields, every n-th entry, probes", group="Output",
                optional=True, of="study_options", stage="solve")
    GROUND = ("x", "y", "z")

    def open_air(type_, method, what):
        return _make(type_, "scene.air", method, [
            _p("ground", "choice", help="the axis normal to the rigid ground", group="Ground", choices=GROUND),
            _p("level", "number", "m", "the ground's coordinate along that axis", "Ground"),
            _p("margin", "number", "m", "hemisphere radius beyond the structure", "DtN hemisphere", min=0),
            _p("bands", "pairs", "Hz", f"frequency bands (f_lo, f_hi): one {what} mesh each, sized for f_hi",
               "Bands"),
            _p("points_per_wavelength", "number", help="elements per wavelength at the band's f_hi", group="Mesh",
               min=0),
            _p("max_size", "number", "m", "largest element", "Mesh", min=0),
            _p("driver_size", "number", "m", "element size on the drivers", "Mesh", min=0),
            _p("near", "pair", "m", "size ramp distances from the drivers (start, end)", "Mesh"),
            _p("plug_from", "number", "m", "local z where the driver plug starts", "Mesh"),
            _p("file", "text", help="mesh file name pattern ({kind}, {f_hi})", group="Files")],
            context=("scene",))

    blocks = [
        # model
        _make("Model", "model", lp.Model, [
            _p("geometry", "file", help="the geometry file (.geo, .step, .stp, .iges, .brep)", group="Geometry"),
            _p("materials", "file", help="a material library file (JSON); empty: loudpy's library",
               group="Materials", optional=True),
            _p("mesh_order", "choice", help="element order: 2 (P2, curved) or 1", group="Mesh", choices=(1, 2)),
            _p("grading", "number", help="element size growth away from small features (empty: gmsh's)",
               group="Mesh", optional=True, min=0),
            _p("parameters", "mapping", help="named values the items refer to with lp.P('name')",
               group="Parameters", optional=True)],
            signature=lp.Model.__init__,
            code_only={"dim": "dim=3 gives a 3D scene: see Scene", "geometry (builder)": "a Python builder function"}),
        # regions
        _make("Solid", "regions", lp.Solid, [
            NAME, ON, MATERIAL,
            _p("damping", "block", help="Rayleigh: fitted to the material's loss factor · Hysteretic · Undamped",
               group="Material", of="damping"),
            MESH,
            RHO, _p("E", "number", "Pa", "Young's modulus; empty: the material's", "Overrides", optional=True, min=0),
            _p("nu", "number", help="Poisson's ratio; empty: the material's", group="Overrides", optional=True,
               min=-1, max=0.5)]),
        _make("Air", "regions", lp.Air, [NAME, ON, MATERIAL, MESH, RHO, C]),
        _make("PML", "regions", lp.PML, [
            NAME, replace(ON_CURVES, help="the base curves it grows from (straight lines r = const or z = const), "
                                          "outward from the air; default: the groups named like it"),
            MATERIAL,
            _p("thickness", "number", "m", "layer thickness (a quarter wavelength at tuned_to is plenty)", "Layer",
               min=0),
            MESH,
            _p("tuned_to", "number", "Hz", "the frequency the absorption is tuned to (empty: the study's)",
               "Absorption", optional=True, min=0),
            _p("order", "number", help="absorption profile: 1 linear, 2 quadratic", group="Absorption", min=0),
            _p("strength", "number", help="absorption strength (larger: shorter decay, more reflection)",
               group="Absorption", min=0),
            RHO, C]),
        # boundaries
        _make("Clamped", "boundaries", lp.Clamped, [
            NAME, ON_CURVES, _p("dofs", "choice", help="rz clamped · r radial only (the axis) · z axial only",
                                group="Condition", choices=("rz", "r", "z"))]),
        _make("Coil", "boundaries", lp.Coil, [
            NAME, ON_CURVES, _p("direction", "choice", help="the coil's force direction", group="Condition",
                                choices=("z", "r")),
            _p("motor", "block", help="the driver's motor: its blocked coil and force factor Bl (a voltage or a "
               "circuit drives it; a scene's driver takes it from the modes file)", group="Motor",
               of="circuit.coils", optional=True)]),
        _make("FluidStructure", "boundaries", lp.FluidStructure, [NAME, ON_CURVES]),
        _make("Radiation", "boundaries", lp.Radiation, [NAME, ON_CURVES]),
        # mesh controls
        _make("MeshSize", "mesh_controls", lp.MeshSize, [
            replace(ON, optional=False, help="the domain groups (or a selector)"),
            _p("size", "number", "m", "element size there (overrides the regions' mesh_size)", "Mesh", min=0),
            replace(NAME, optional=True)]),
        # materials
        _make("Material", "materials", lp.Material, [
            replace(NAME, help="its name in the library"),
            _p("kind", "choice", help="solid or fluid", group="Identity", choices=("solid", "fluid")),
            _p("rho", "number", "kg/m³", "density", "Properties", optional=True, min=0),
            _p("E", "number", "Pa", "Young's modulus (solid)", "Properties", optional=True, min=0),
            _p("nu", "number", help="Poisson's ratio (solid)", group="Properties", optional=True, min=-1, max=0.5),
            _p("c", "number", "m/s", "speed of sound (fluid)", "Properties", optional=True, min=0),
            _p("rayleigh", "pair", help="default Rayleigh fit (alpha [1/s], beta [s]): C = alpha M + beta K",
               group="Damping", optional=True),
            _p("loss_factor", "number", help="default hysteretic loss factor eta (solid)", group="Damping",
               optional=True, min=0, min_inclusive=True),
            _p("temperature", "number", "K", "reference temperature of the values", "Conditions", optional=True,
               min=0),
            _p("pressure", "number", "Pa", "static pressure (a gas from lp.air())", "Conditions", optional=True,
               min=0),
            _p("humidity", "number", help="relative humidity 0–1 (a gas from lp.air())", group="Conditions",
               optional=True, min=0, min_inclusive=True, max=1, max_inclusive=True),
            _p("source", "text", help="where the values come from", group="Identity")]),
        # damping
        _make("Rayleigh", "damping", lp.Rayleigh, [
            _p("alpha", "number", "1/s", "mass-proportional term (empty: the material's fit)", optional=True),
            _p("beta", "number", "s", "stiffness-proportional term (empty: the material's fit)", optional=True)]),
        _make("Hysteretic", "damping", lp.Hysteretic, [
            _p("eta", "number", help="loss factor: K (1 + i eta) (empty: the material's)", optional=True, min=0,
               min_inclusive=True)]),
        _make("Undamped", "damping", lp.Undamped, []),
        # drives
        _make("Force", "drives", lp.Force, [
            _p("amplitude", "complex", "N", "harmonic: a (complex) amplitude; transient: the force signal",
               "Drive"),
            _p("fs", "number", "Hz", "sampling rate of a force signal (transient)", "Drive", optional=True, min=0)]),
        _make("Voltage", "drives", lp.Voltage, [
            _p("amplitude", "complex", "V", "voltage amplitude on the coil", "Drive"),
            _p("Re", "number", "Ω", "coil resistance (empty: the driver's motor)", "Coil", min=0, optional=True),
            _p("Bl", "number", "T·m", "force factor (empty: the driver's motor)", "Coil", optional=True),
            _p("Le", "number", "H", "coil (semi-)inductance", "Coil", min=0, min_inclusive=True),
            _p("n", "number", help="semi-inductance exponent: Z_L = Le (jω)^n (1: a plain inductance)",
               group="Coil", min=0, max=1, max_inclusive=True)]),
        _make("CircuitRef", "drives", _circuit_ref, [
            _p("name", "text", help="one of the simulation's circuits (lp.Document: circuits)", group="Circuit")],
            context=("circuits",)),
        _make("Circuit", "drives", lp.Circuit, [
            _p("parts", "blocks", help="the amplifier, filter blocks, passive parts and the drivers' coils, on named "
               "nodes ('0': the ground)", group="Circuit", of="circuit.parts")], varargs="parts"),
        # a circuit's parts (loudpy.circuit)
        _make("circuit.Source", "circuit.parts", circ.Source, [
            _p("name", "text", help="its name", group="Part"),
            _p("a", "text", help="its + node", group="Nodes"),
            _p("b", "text", help="its − node ('0': the ground)", group="Nodes"),
            _p("U", "complex", "V", "the amplifier's voltage (a complex amplitude)", "Amplifier"),
            _p("R_out", "number", "Ω", "its output resistance", "Amplifier", min=0, min_inclusive=True)]),
        _make("circuit.Filter", "circuit.parts", circ.Filter, [
            _p("name", "text", help="its name", group="Part"),
            _p("inp", "text", help="the node it reads (to ground)", group="Nodes"),
            _p("out", "text", help="the node it drives (to ground)", group="Nodes"),
            _p("H", "blocks", help="its stages, cascaded: low / high pass, EQ, shelving, gain, delay",
               group="Filter", of="circuit.filters")]),
        _make("circuit.Resistor", "circuit.parts", circ.Resistor, [
            _p("name", "text", help="its name", group="Part"), _p("a", "text", help="a node", group="Nodes"),
            _p("b", "text", help="the other node", group="Nodes"),
            _p("R", "number", "Ω", "its resistance", "Value", min=0)]),
        _make("circuit.Capacitor", "circuit.parts", circ.Capacitor, [
            _p("name", "text", help="its name", group="Part"), _p("a", "text", help="a node", group="Nodes"),
            _p("b", "text", help="the other node", group="Nodes"),
            _p("C", "number", "F", "its capacitance", "Value", min=0)]),
        _make("circuit.Inductor", "circuit.parts", circ.Inductor, [
            _p("name", "text", help="its name", group="Part"), _p("a", "text", help="a node", group="Nodes"),
            _p("b", "text", help="the other node", group="Nodes"),
            _p("L", "number", "H", "its inductance", "Value", min=0)]),
        _make("circuit.Coil", "circuit.parts", circ.Coil, [
            _p("name", "text", help="its name", group="Part"), _p("a", "text", help="its + node", group="Nodes"),
            _p("b", "text", help="its − node", group="Nodes"),
            _p("driver", "text", help="the scene's driver it moves (empty: a 2D model's coil)", group="Coil",
               optional=True),
            _p("model", "block", help="its motor here (empty: the driver's motor)", group="Coil",
               of="circuit.coils", optional=True)]),
        # a filter's stages
        _make("circuit.butterworth", "circuit.filters", circ.butterworth, [
            _p("order", "integer", help="1, 2, 3 …", min=0), _p("fc", "number", "Hz", "−3 dB frequency", min=0),
            _p("kind", "choice", help="low or high pass", choices=("low", "high"))]),
        _make("circuit.linkwitz_riley", "circuit.filters", circ.linkwitz_riley, [
            _p("order", "integer", help="an even order: 2, 4, 8", min=0),
            _p("fc", "number", "Hz", "crossover frequency (−6 dB)", min=0),
            _p("kind", "choice", help="low or high pass", choices=("low", "high"))]),
        _make("circuit.peq", "circuit.filters", circ.peq, [
            _p("f0", "number", "Hz", "centre frequency", min=0), _p("gain_db", "number", "dB", "gain at f0"),
            _p("q", "number", help="quality factor", min=0)]),
        _make("circuit.low_shelf", "circuit.filters", circ.low_shelf, [
            _p("f0", "number", "Hz", "corner frequency", min=0), _p("gain_db", "number", "dB", "gain below"),
            _p("q", "number", help="slope (0.707: Butterworth)", min=0)]),
        _make("circuit.high_shelf", "circuit.filters", circ.high_shelf, [
            _p("f0", "number", "Hz", "corner frequency", min=0), _p("gain_db", "number", "dB", "gain above"),
            _p("q", "number", help="slope (0.707: Butterworth)", min=0)]),
        _make("circuit.gain", "circuit.filters", circ.gain, [_p("db", "number", "dB", "gain")]),
        _make("circuit.delay", "circuit.filters", circ.delay, [_p("t", "number", "s", "delay", min=0,
                                                                   min_inclusive=True)]),
        _make("circuit.biquad", "circuit.filters", circ.biquad, [
            _p("num", "numbers", help="numerator coefficients (highest power of s first)"),
            _p("den", "numbers", help="denominator coefficients (highest power of s first)")]),
        # coil models (the motor)
        _make("circuit.Semi", "circuit.coils", circ.Semi, [
            _p("Re", "number", "Ω", "resistance", min=0), _p("Bl", "number", "T·m", "force factor"),
            _p("Le", "number", "H", "(semi-)inductance", min=0, min_inclusive=True),
            _p("n", "number", help="exponent: Le (jω)^n (1: an inductance)", min=0, max=1, max_inclusive=True)]),
        _make("circuit.Leach", "circuit.coils", circ.Leach, [
            _p("Re", "number", "Ω", "resistance", min=0), _p("Bl", "number", "T·m", "force factor"),
            _p("K", "number", help="Leach's K", min=0, min_inclusive=True),
            _p("n", "number", help="Leach's exponent", min=0, max=1, max_inclusive=True)]),
        _make("circuit.LR2", "circuit.coils", circ.LR2, [
            _p("Re", "number", "Ω", "resistance", min=0), _p("Bl", "number", "T·m", "force factor"),
            _p("Le", "number", "H", "inductance", min=0, min_inclusive=True),
            _p("R2", "number", "Ω", "eddy-current resistance", min=0, min_inclusive=True),
            _p("L2", "number", "H", "eddy-current inductance", min=0, min_inclusive=True)]),
        _make("circuit.Wright", "circuit.coils", circ.Wright, [
            _p("Re", "number", "Ω", "resistance", min=0), _p("Bl", "number", "T·m", "force factor"),
            _p("Krm", "number", help="resistive coefficient", min=0, min_inclusive=True),
            _p("Erm", "number", help="resistive exponent"),
            _p("Kxm", "number", help="reactive coefficient", min=0, min_inclusive=True),
            _p("Exm", "number", help="reactive exponent")]),
        # selectors
        _make("Group", "selectors", sel.Group, [
            _p("labels", "texts", help="names of physical groups of the geometry")], varargs="labels"),
        _make("Box", "selectors", sel.Box, [
            _p("r", "range", "m", "r = value, or min ≤ r ≤ max (2D)", optional=True),
            _p("z", "range", "m", "z = value, or min ≤ z ≤ max (2D)", optional=True),
            _p("x", "range", "m", "x = value, or a range (3D)", optional=True),
            _p("y", "range", "m", "y = value, or a range (3D)", optional=True),
            _p("tol", "number", "m", "tolerance (empty: relative to the geometry)", optional=True, min=0)]),
        _make("Near", "selectors", sel.Near, [_p("point", "point", "m", "the entity nearest this point")]),
        _make("Interface", "selectors", sel.Interface, [
            _p("a", "selection", help="one side"), _p("b", "selection", help="the other side")]),
        _make("BoundaryOf", "selectors", sel.BoundaryOf, [_p("a", "selection", help="the domains")]),
        _make("Union", "selectors", sel._Union, [_p("a", "selection", help="a selection (either)"),
                                                       _p("b", "selection", help="another selection")],
              code_only={"syntax": "a | b"}),
        _make("Intersection", "selectors", sel._Intersection, [_p("a", "selection", help="a selection (both)"),
                                                       _p("b", "selection", help="another selection")],
              code_only={"syntax": "a & b"}),
        _make("Difference", "selectors", sel._Difference, [_p("a", "selection", help="a selection (the first without the second)"),
                                                       _p("b", "selection", help="another selection")],
              code_only={"syntax": "a - b"}),
        _make("Item", "selectors", sel._Item, [], context=("item",),
              code_only={"syntax": "a model item (region or boundary) used as a selector: what it covers"}),
        # profiles (a driver's wet surfaces)
        _make("Profile", "profiles", lp.Profile, [
            _p("on", "selection", help="the curves of the driver's 2D geometry (a selector, or gmsh curve tags)"),
            _p("start", "point", "m", "(r, z) on the baffle where the profile starts (empty: the chain's end "
               "farthest from the axis)", optional=True),
            _p("end", "point", "m", "(r, z) where it ends (empty: the chain's other end)", optional=True)]),
        # studies
        _make("Modes", "studies", lp.Modes, [
            _p("physics", "choice", help="structure · acoustic · coupled (2D) · walls (a scene's elastic walls)",
               group="Study", choices=("structure", "acoustic", "coupled", "walls")),
            _p("f_max", "number", "Hz", "highest mode frequency", "Frequencies", min=0, stage="solve"),
            _p("f_min", "number", "Hz", "lowest mode frequency", "Frequencies", min=0, min_inclusive=True,
               stage="solve"),
            _p("orders", "integers", help="azimuthal orders m (empty: the axisymmetric modes, m = 0)",
               group="Frequencies", optional=True, stage="solve"),
            _p("method", "choice", help="multishift: several shift-invert solves over the band · arpack: one solve, "
               "n_modes around f_target (without orders)", group="Solver", choices=("multishift", "arpack"),
               stage="solve", when=((("physics", ("structure",)),),)),
            _p("n_shifts", "integer", help="number of shifts, log-spaced over f_min … f_max (empty: about one per "
               "octave)", group="Solver", min=0, optional=True, stage="solve",
               when=((("physics", ("structure",)), ("method", ("multishift",))),)),
            _p("n_per_shift", "integer", help="modes computed around each shift", group="Solver", min=0,
               stage="solve", when=((("physics", ("structure",)), ("method", ("multishift",))),
                                    (("physics", ("coupled",)),))),
            _p("n_modes", "integer", help="modes computed around f_target", group="Solver", min=0, stage="solve",
               when=((("physics", ("structure",)), ("method", ("arpack",))), (("physics", ("acoustic",)),))),
            _p("f_target", "number", "Hz", "the frequency the modes are sought around", "Solver", min=0,
               stage="solve", when=((("physics", ("structure",)), ("method", ("arpack",))),
                                    (("physics", ("acoustic",)),))),
            OUTPUT],
            context=("model",), solve=lp.Modes.solve,
            code_only={"options": "solver options (f_targets, mac_tol, coupled-mode covering …): see Modes.solve"}),
        _make("FrequencyResponse", "studies", lp.FrequencyResponse, [
            _p("drive", "block", help="Force or Voltage on the coil (a scene: on every driver's coil, or one per "
               "driver)",
               group="Drive", of="drives", optional=True),
            _p("coupled", "bool", help="structure + air (empty: coupled when the model has air)", group="Study",
               optional=True),
            _p("reuse_factorization", "bool", help="reuse the factorization between close frequencies",
               group="Solver"),
            _p("freqs", "frequencies", "Hz", "the frequencies", "Frequencies", stage="solve"),
            _p("remesh", "block", help="air and PML meshes per frequency band (PerWavelength)", group="Mesh",
               of="study_options", optional=True, stage="solve"),
            _p("store", "file", help="a scene's result file the 3D field is streamed to", group="Output",
               optional=True, stage="solve"),
            OUTPUT],
            context=("model",), solve=lp.FrequencyResponse.solve,
            code_only={"options": "solver options: see FrequencyResponse.solve",
                       "drive (per driver)": "a scene's {driver name: Force}"}),
        _make("Transient", "studies", lp.Transient, [
            _p("drive", "block", help="a Force signal on the coil", group="Drive", of="drives"),
            _p("nonlinear", "bool", help="the coil's geometric nonlinearity (Newton–Raphson)", group="Solver"),
            _p("rho_inf", "number", help="generalised-α high-frequency dissipation (0–1)", group="Solver", min=0,
               min_inclusive=True, max=1, max_inclusive=True),
            _p("nr_tol", "number", help="Newton–Raphson tolerance", group="Solver", min=0),
            _p("nr_max_iter", "integer", help="Newton–Raphson iterations at most", group="Solver", min=0),
            _p("n_steps", "integer", help="time steps (or a duration)", group="Time", optional=True, min=0,
               stage="solve"),
            _p("duration", "number", "s", "simulated time (or n_steps)", "Time", optional=True, min=0,
               stage="solve"),
            OUTPUT],
            context=("model",), solve=lp.Transient.solve),
        _make("Sweep", "studies", lp.Sweep, [
            _p("parameters", "mapping", help="{parameter: values} (or one name, with values)", group="Sweep"),
            _p("values", "numbers", help="the values of a single swept parameter", group="Sweep", optional=True),
            _p("mesh_folder", "file", help="where each run's mesh goes", group="Files", optional=True),
            _p("combine", "choice", help="grid: every combination · zip: the i-th values together", group="Sweep",
               choices=("grid", "zip")),
            _p("redrive", "bool", help="parameters of the circuit only: one solve, the runs re-driven (empty: "
               "when it applies)", group="Sweep", optional=True)],
            context=("study",), code_only={"solve arguments": "those of the wrapped study"}),
        _make("Adaptive", "studies", lp.Adaptive, [
            _p("target", "number", help="relative error estimate to reach", group="Adaptive", min=0, max=1),
            _p("max_iterations", "integer", help="adaptation rounds at most (0: the first mesh only)",
               group="Adaptive", min=0, min_inclusive=True),
            _p("folder", "file", help="where the adapted meshes go", group="Files", optional=True)],
            context=("study",), code_only={"solve arguments": "those of the wrapped study"}),
        # study options
        _make("PerWavelength", "study_options", lp.PerWavelength, [
            _p("bands", "numbers", "Hz", "frequencies where a new band (and mesh) starts", "Bands"),
            _p("every", "integer", help="or: a new mesh every `every` frequencies", group="Bands", optional=True,
               min=0),
            _p("air_elements", "number", help="air elements per wavelength λ = c / f0", group="Air", min=0),
            _p("air_max_size", "number", "m", "largest air element (empty: the Air region's mesh_size)", "Air",
               optional=True, min=0),
            _p("pml_thickness", "number", help="PML thickness in wavelengths", group="PML", min=0),
            _p("pml_elements", "number", help="elements across the PML", group="PML", min=0),
            _p("c", "number", "m/s", "speed of sound for λ (empty: the Air material's)", "Air", optional=True,
               min=0),
            _p("folder", "file", help="where the band meshes go (empty: next to the geometry)", group="Files",
               optional=True),
            _p("size_at", "choice", help="top: air elements per wavelength at the band's top frequency · start: at its "
               "first frequency (meshes made before 0.2.2)", group="Air", choices=("top", "start"))]),
        _make("Output", "study_options", lp.Output, [
            _p("fields", "texts", help="fields kept (empty: all)", optional=True),
            _p("every", "integer", help="keep every n-th entry", min=0),
            _p("probes", "mapping", "m", "{name: point}: recorded at every entry", optional=True)]),
        # transient signals
        _make("signals.sine", "signals", lp.signals.sine, [
            _p("f", "number", "Hz", "frequency", min=0),
            _p("periods", "number", help="periods (with a raised-cosine start)", min=0),
            _p("fs", "number", "Hz", "sampling rate", min=0),
            _p("amplitude", "number", help="peak value (N for a Force)"),
            _p("duration", "number", "s", "length (empty: periods / f)", optional=True, min=0)]),
        _make("signals.multitone", "signals", lp.signals.multitone, [
            _p("freqs", "numbers", "Hz", "the tones (Schroeder phases)"),
            _p("duration", "number", "s", "length (empty: periods of the lowest tone)", optional=True, min=0),
            _p("fs", "number", "Hz", "sampling rate", min=0),
            _p("amplitude", "number", help="peak value (N for a Force)"),
            _p("periods", "number", help="periods of the lowest tone", min=0)]),
        _make("signals.wav", "signals", lp.signals.wav, [
            _p("path", "file", help="a WAV file"),
            _p("duration", "number", "s", "length kept (empty: all)", optional=True, min=0),
            _p("amplitude", "number", help="peak value after scaling (N for a Force)")]),
        # a 3D scene
        _make("Scene", "scene", lp.Scene, [
            _p("cabinet", "block", help="a parametric Box or a CAD (STEP) cabinet", group="Cabinet",
               of="scene.cabinet"),
            _p("air", "material", help="the air's material", group="Air"),
            _p("parameters", "mapping", help="named values the cabinet and settings refer to (lp.P)",
               group="Parameters", optional=True)],
            context=("drivers",),
            code_only={"cabinet (builder)": "a Python function returning an lp.Cabinet (gmsh OCC)"}),
        _make("cabinets.Box", "scene.cabinet", cab.Box, [
            _p("width", "number", "m", "inner width (x)", "Size", min=0),
            _p("height", "number", "m", "inner height (y, from bottom)", "Size", min=0),
            _p("depth", "number", "m", "inner depth (z, back to front, at the bottom)", "Size", min=0),
            _p("depth_top", "number", "m", "depth at the top (a slanted back; empty: depth)", "Size",
               optional=True, min=0),
            _p("wall", "number", "m", "thickness of the sides, top and back (0: the inside up to the front)",
               "Walls", min=0, min_inclusive=True),
            _p("bottom", "number", "m", "thickness of the bottom (0: the inside air stands on the ground)",
               "Walls", min=0, min_inclusive=True),
            _p("braces", "blocks", help="horizontal shelves across the width, from the back", group="Inside",
               of="scene.cabinet_parts"),
            _p("ports", "blocks", help="round tubes through the front", group="Inside", of="scene.cabinet_parts"),
            _p("port_size", "number", "m", "mesh size on the port walls (empty: the default sizes)", "Mesh",
               optional=True, min=0),
            _p("front", "number", "m", "z of the inner front plane (empty: the drivers' baffle plane)", "Size",
               optional=True),
            _p("front_face", "number", "m", "z of the outer front face (empty: the drivers' front plane)",
               "Size", optional=True)]),
        _make("cabinets.Step", "scene.cabinet", cab.Step, [
            _p("path", "file", help="the cabinet as built: STEP / IGES / BREP, any unit (read in metres)",
               group="File"),
            _p("openings", "blocks", help="port bores: a Port, or a point inside the bore's solid", group="Inside",
               of="scene.cabinet_parts"),
            _p("inner", "point", "m", "a point inside the cabinet's air (empty: the closed void found)",
               "Inside", optional=True),
            _p("scale", "number", help="scale applied to the file", group="Placement", min=0),
            _p("translate", "point", "m", "translation applied to the file", "Placement"),
            _p("port_size", "number", "m", "mesh size on the port walls (empty: the default sizes)", "Mesh",
               optional=True, min=0),
            _p("plug_drivers", "bool", help="close the driver holes between their baffle and front planes",
               group="Inside"),
            _p("ground", "number", "m", "y of the ground that closes a box without bottom (empty: the file is "
               "closed)", "Placement", optional=True),
            _p("drivers", "blocks", help="the drivers the file was drawn for (hole checks)", group="Inside",
               of="scene.drivers")]),
        _make("cabinets.Brace", "scene.cabinet_parts", cab.Brace, [
            _p("y", "number", "m", "height from the inner bottom"),
            _p("thickness", "number", "m", "shelf thickness", min=0),
            _p("depth", "number", "m", "depth from the back", min=0)]),
        _make("cabinets.Port", "scene.cabinet_parts", cab.Port, [
            _p("at", "point", "m", "(x, y) of the port's axis on the front face"),
            _p("radius", "number", "m", "bore radius", min=0),
            _p("length", "number", "m", "length from the front face inwards", min=0),
            _p("wall", "number", "m", "tube wall thickness inside the box (0: a plain hole)", min=0,
               min_inclusive=True)]),
        _make("Driver.from_modes", "scene.drivers", lp.Driver.from_modes, [
            _p("path", "file", help="the eigen result file of the driver's axisymmetric model", group="Modes"),
            _p("orders", "integers", help="azimuthal orders kept (0: piston-like, 1: rocking)", group="Modes"),
            _p("f_max", "number", "Hz", "modes kept up to (empty: all in the file)", "Modes", optional=True,
               min=0),
            _p("geometry", "file", help="the driver's 2D geometry (or its model)", group="Geometry"),
            _p("rear", "block", help="the wet profile on the cabinet side", group="Wet surfaces", of="profiles"),
            _p("front", "block", help="the wet profile on the outside (empty: the rear one)",
               group="Wet surfaces", of="profiles", optional=True),
            _p("coil", "text", help="the coil boundary's name (empty: the model's lp.Coil item)", group="Coil",
               optional=True),
            _p("motor", "block", help="its motor (empty: the one its modes file describes, its lp.Coil item's)",
               group="Coil", of="circuit.coils", optional=True),
            _p("name", "text", help="the driver's name (empty: the file's)", group="Identity", optional=True)],
            code_only={"geometry (builder or model)": "a builder function or its lp.Model"}),
        _make("Driver.place", "scene.placement", lp.Driver.place, [
            NAME,
            _p("at", "point", "m", "where its axis meets the front: (x, y) on the front face, or (x, y, z)",
               "Placement"),
            _p("axis", "point", help="its axis direction (empty: +z, facing the front)", group="Placement",
               optional=True),
            _p("ref", "point", help="a direction fixing its rotation about the axis (empty: any)",
               group="Placement", optional=True)],
            context=("driver",)),
        _make("Scene.cavity", "scene.air", lp.Scene.cavity, [
            _p("mesh_size", "pair", "m", "element size (near the drivers, far)", "Mesh"),
            _p("f_max", "number", "Hz", "cavity modes kept up to", "Modes", min=0),
            _p("eta", "number", help="modal loss factor of the cavity air", group="Modes", min=0,
               min_inclusive=True),
            _p("near", "pair", "m", "size ramp distances from the drivers (start, end)", "Mesh"),
            _p("file", "text", help="the cavity mesh file name", group="Files"),
            _p("cache_modes", "bool", help="keep the cavity modes next to the mesh", group="Files")],
            context=("scene",)),
        _make("Scene.walls", "scene.walls", lp.Scene.walls, [
            _p("material", "material", help="the walls' material (solid)", group="Material"),
            _p("thickness", "number", "m", "shell thickness (empty: each panel's, measured on the Box / STEP "
               "cabinet)", "Material", min=0, optional=True),
            _p("damping", "block", help="empty: Rayleigh fitted to the material", group="Material",
               of="damping", optional=True),
            _p("fixed", "choice", help="edges: the panel edges and the driver cut-out rims are clamped",
               group="Supports", choices=("edges",)),
            _p("f_max", "number", "Hz", "wall modes kept up to (empty: the cavity's)", "Modes", optional=True,
               min=0),
            _p("angle", "number", "°", "a fold sharper than this is a panel edge", "Supports", min=0),
            _p("panels", "mapping", help="panels of their own: {a STEP solid's index (0, 1 …) or a Box side "
               "(left, right, bottom, top, back, front): a Panel}", group="Panels", optional=True),
            _p("mesh_size", "pair", "m", "one merged air domain: element size of the inside-air mesh the shells lie "
               "on (near the drivers, far; empty: 3 mm, 12 mm); with cavity modes, the cavity's", "Mesh",
               optional=True)],
            context=("scene",), code_only={"fixed (function)": "f(points) → mask of the clamped nodes"}),
        _make("cabinets.Panel", "scene.panels", cab.Panel, [
            _p("material", "material", help="its material (empty: the walls')", optional=True),
            _p("thickness", "number", "m", "its thickness (empty: the walls', else measured on the cabinet)",
               optional=True, min=0),
            _p("damping", "block", help="its damping (empty: the walls')", of="damping", optional=True)]),
        _make("Scene.keep", "scene.air", lp.Scene.keep, [
            _p("meshes", "texts", help="meshes used as they are while their file exists, even out of date: cavity, "
               "walls, air (every band), air <top frequency>, all")],
            context=("scene",)),
        open_air("Scene.exterior", lp.Scene.exterior, "exterior"),
        open_air("Scene.merged", lp.Scene.merged, "air"),
    ]
    return {b.type: b for b in blocks}


_CATALOG: dict[str, Block] | None = None
SECTIONS = ("model", "materials", "regions", "boundaries", "mesh_controls", "damping", "drives",
            "selectors", "profiles", "studies", "study_options", "signals", "scene", "scene.cabinet",
            "scene.cabinet_parts", "scene.drivers", "scene.placement", "scene.air", "scene.walls", "scene.panels",
            "circuit.parts", "circuit.filters", "circuit.coils")

# Public names of loudpy that are not building blocks of a simulation, and why (the completeness test reads this).
NOT_BLOCKS = {
    "Parameters": "the model's `parameters` mapping (Model / Scene parameter)",
    "P": "a value: a reference to a parameter ({'P': name})",
    "MaterialLibrary": "a collection: its materials are Material blocks",
    "materials": "loudpy's default material library (a collection)",
    "load_materials": "reads a library file (the Model's `materials`)",
    "air": "computes an air Material from conditions: described as the Material it returns",
    "Frame": "a driver's placement, made by Driver.place",
    "Driver": "made by Driver.from_modes (and placed by Driver.place)",
    "Cabinet": "what a Python cabinet builder returns: code only (cabinets.Box / Step are the blocks)",
    "Mesh": "the result of meshing",
    "mesh_info": "reads a mesh file (what it is for, its counts)",
    "Convergence": "the result of an Adaptive study", "convergence": "the result of an Adaptive study",
    "Results": "results", "FrequencyResults": "results", "ModeResults": "results", "TimeResults": "results",
    "SweepResults": "results", "Data": "results", "Field": "results",
    "Point": "evaluation of results", "Points": "evaluation of results", "Polar": "evaluation of results",
    "Arc": "evaluation of results", "Line": "evaluation of results", "On": "evaluation of results",
    "spl": "evaluation of results",
    "load": "reads results", "load_data": "reads results", "compare": "compares results",
    "read_comsol": "reads COMSOL tables", "set_verbosity": "logging",
    "cabinets": "module: cabinets.Box / Step / Brace / Port are blocks", "signals": "module: signals.* are blocks",
    "errors": "module", "progress": "module", "plot": "module", "schema": "module: this description",
    "Document": "a whole simulation (model or scene, and studies) as nodes of these blocks",
    "docpath": "module: paths into a document's data", "readiness": "what a simulation still needs (a checklist)",
    "suggest": "what each geometry group could be (the item types that fit it)",
    "export": "writes results, data, views and figures to files by suffix", "result_path": "where a result goes",
    "Workspace": "a simulation's folder and document, edited by commands",
    "circuit": "module: circuit.Source / Filter / Coil …, its filter stages and coil models are blocks",
}


def catalog() -> dict[str, Block]:
    """{type: Block} of every building block."""
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _catalog()
    return _CATALOG


def item_params(cls) -> list[Param]:
    """The parameters every item has: its name (a label) and `on` (the domain groups or curves it selects, by its
    facts)."""
    from loudpy.model.facts import facts_of
    on = _p("on", "selection", help="the groups (or a selector) it applies to; default: the groups named like it",
            group="Selection", optional=True)
    if facts_of(cls).selects == "curves":
        on = replace(on, help="the curves (groups or a selector) it applies to; default: the groups named like it")
    return [_p("name", "text", help="a label: nothing physical is read from it", group="Identity"), on]


def register(cls, params=(), *, type_: str | None = None, section: str | None = None) -> Block:
    """Describe a new item type (a Region or a Boundary subclass declaring its facts: loudpy.model.facts) and make it
    known: it then shows in the interface's Physics tables, in lp.suggest and in lp.readiness. params: its own
    parameters (lp.schema.Param; its name and `on` are added). A script writes it as `lp.<type>(…)`: a type outside
    loudpy is for programs (the interface, a document), or give loudpy the class (setattr(lp, name, cls)).

        lp.schema.register(Absorber, [lp.schema.Param("strength", "number", help="…", group="Layer", min=0)])
    """
    from loudpy.model import facts as _facts
    from loudpy.model.boundaries import Boundary
    from loudpy.model.regions import Region
    if not (inspect.isclass(cls) and issubclass(cls, (Region, Boundary))):
        raise SchemaError(f"register takes a Region or a Boundary subclass, got {cls!r}")
    _facts.register(cls)
    params = list(params)
    have = {p.name for p in params}
    params = [p for p in item_params(cls) if p.name not in have] + params
    sec = section or ("regions" if issubclass(cls, Region) else "boundaries")
    b = _make(type_ or cls.__name__, sec, cls, params)
    catalog()[b.type] = b
    return b


def unregister(type_: str) -> None:
    """Forget a type made known by register."""
    from loudpy.model import facts as _facts
    b = catalog().pop(type_, None)
    if b is not None:
        _facts.unregister(getattr(b.target, "__name__", type_))


def block(type_: str) -> Block:
    """The Block of a type ("Solid", "cabinets.Box", "Scene.cavity" …)."""
    try:
        return catalog()[type_]
    except KeyError:
        raise SchemaError(f"no building block {type_!r}; known: {sorted(catalog())}") from None


def blocks(section: str | None = None) -> list[Block]:
    """The blocks of a section (all of them without one), in catalog order."""
    if section is not None and section not in SECTIONS:
        raise SchemaError(f"no section {section!r}; sections: {list(SECTIONS)}")
    return [b for b in catalog().values() if section is None or b.section == section]


# ── values ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def to_value(x) -> Any:
    """A value as plain data (JSON-able): numbers, strings, lists, None; lp.P as {"P": …}; complex as {"complex":
    [re, im]}; a building block as its node."""
    from loudpy.model.parameters import P
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return x
    if isinstance(x, np.generic):
        return to_value(x.item())
    if isinstance(x, complex):
        return {"complex": [x.real, x.imag]} if x.imag else x.real
    if isinstance(x, np.ndarray):
        return [to_value(v) for v in x.tolist()]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, P):
        out: dict = {"P": x.name}
        if x.scale != 1.0:
            out["scale"] = x.scale
        if x.offset != 0.0:
            out["offset"] = x.offset
        return out
    if isinstance(x, (list, tuple)):
        return [to_value(v) for v in x]
    if isinstance(x, dict):
        return {str(k): to_value(v) for k, v in x.items()}
    from loudpy.circuit import Transfer
    if isinstance(x, Transfer):                                       # a filter's H: the stages it was made of
        return [{"type": p["type"], "params": to_value(p["params"])} for p in x.parts]
    return describe(x)


def from_value(v, p: Param | None = None, **context) -> Any:
    """The object a plain value stands for (to_value's inverse), shaped for the parameter p (tuples where the code
    keeps tuples)."""
    from loudpy.model.parameters import P
    if isinstance(v, dict):
        if p is not None and p.kind == "frequencies" and "from" in v:
            return frequencies(v)
        if "type" in v:
            return build(v, **context)
        if "P" in v:
            return P(v["P"], float(v.get("scale", 1.0)), float(v.get("offset", 0.0)))
        if "complex" in v:
            re, im = v["complex"]
            return complex(re, im)
        return {k: from_value(x, None, **context) for k, x in v.items()}
    if isinstance(v, list):
        items = [from_value(x, None, **context) for x in v]
        if p is not None and p.kind == "pairs":
            items = [tuple(i) if isinstance(i, list) else i for i in items]
        tuple_ = p is not None and (p.kind in _TUPLE_KINDS or isinstance(p.default, tuple))
        return tuple(items) if tuple_ else items
    return v


def frequencies(v) -> np.ndarray:
    """The frequencies of a "frequencies" value: a list, or {"from", "to", "points", "spacing": "log" | "linear"}."""
    if not isinstance(v, dict):
        return np.asarray(v, float)
    a, b, n = float(v["from"]), float(v["to"]), int(v.get("points", 100))
    if n < 1 or a <= 0 or b < a:
        raise SchemaError(f"frequencies: 0 < from ≤ to and points ≥ 1, got {v!r}")
    return np.geomspace(a, b, n) if v.get("spacing", "log") == "log" else np.linspace(a, b, n)


# ── nodes ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def defaults(type_: str) -> dict:
    """A node of this type with its defaults (required parameters: a placeholder of their kind)."""
    b = block(type_)
    placeholder = {"text": type_.split(".")[-1].lower(), "number": 0.0, "file": "", "selection": "",
                   "material": "Air" if b.facts is not None and b.facts.physics == "air" else "", "texts": [],
                   "numbers": [], "integers": [],
                   "point": [0.0, 0.0], "complex": 1.0, "mapping": {}, "block": None}
    params = {}
    for p in b.params:
        if p.stage == "solve" and not p.required:
            params[p.name] = to_value(p.default)
        elif p.required:
            params[p.name] = placeholder.get(p.kind)
        else:
            params[p.name] = to_value(p.default)
    return {"type": b.type, "params": params}


_SELECTOR_TYPES = {"_Union": "Union", "_Intersection": "Intersection", "_Difference": "Difference", "_Item": "Item"}


def _type_of(obj) -> str:
    name = type(obj).__name__
    mod = type(obj).__module__
    if name in _SELECTOR_TYPES:
        return _SELECTOR_TYPES[name]
    if mod.endswith("scene.cabinets"):
        return f"cabinets.{name}"
    if mod == "loudpy.circuit" and name != "Circuit":
        return f"circuit.{name}"
    return name


def describe(obj) -> dict:
    """The node of a building block object (its parameters that differ from the defaults; required ones always).
    Studies, scenes, models and drivers are described by the simulation document, not here."""
    t = _type_of(obj)
    b = catalog().get(t)
    if b is None or b.section in ("studies", "scene", "model", "signals", "scene.air", "scene.walls",
                                  "scene.drivers", "scene.placement"):
        raise SchemaError(f"cannot describe a {type(obj).__name__} as a node (not a value block)")
    if t == "Group":
        return {"type": t, "params": {"labels": list(obj.labels_)}}
    if t == "Box":
        vals = {k: (list(v) if v[0] != v[1] else v[0]) for k, v in obj.bounds.items()}
        if obj.tol is not None:
            vals["tol"] = obj.tol
        return {"type": t, "params": vals}
    if t == "Near":
        return {"type": t, "params": {"point": to_value(np.asarray(obj.point))}}
    if t == "Item":
        return {"type": t, "params": {"name": obj.item.name}}
    bound = {f.name for f in dataclasses.fields(obj) if not f.compare} if dataclasses.is_dataclass(obj) else set()
    params = {}
    for p in b.params:
        if p.name in bound:                                          # set by its surroundings (a Step's drivers)
            continue
        v = getattr(obj, p.name)
        if p.required or not _same(v, p.default):
            params[p.name] = to_value(v)
    return {"type": t, "params": params}


def _same(a, b) -> bool:
    try:
        return bool(np.all(to_value(a) == to_value(b))) if not isinstance(to_value(a), (dict, list)) else \
            to_value(a) == to_value(b)
    except SchemaError:
        return a is b


def build(node: dict, **context):
    """The object of a node (a study: (study, solve kwargs); context: what its block's context names — the model a
    study runs on, the study a Sweep wraps, the scene a setting applies to, the items an Item selects by name)."""
    if not isinstance(node, dict) or "type" not in node:
        raise SchemaError(f"a node is {{'type': …, 'params': {{…}}}}, got {node!r}")
    b = block(node["type"])
    given = dict(node.get("params") or {})
    unknown = set(given) - {p.name for p in b.params}
    if b.type == "Item":
        unknown -= {"name"}
    if unknown:
        raise SchemaError(f"{b.type}: unknown parameter(s) {sorted(unknown)}; parameters: "
                          f"{[p.name for p in b.params]}")
    if b.type == "Item":
        items = context.get("items") or {}
        if given.get("name") not in items:
            raise SchemaError(f"Item {given.get('name')!r}: give the model's items (build(node, items={{name: "
                              "item}}))")
        return b.target(items[given["name"]])
    values = {p.name: from_value(given[p.name], p, **context) for p in b.params if p.name in given}
    missing = [p.name for p in b.params if p.required and p.name not in values]
    if missing:
        raise SchemaError(f"{b.type}: missing {missing}")
    args = [context[c] for c in b.context if c in context]
    missing_ctx = [c for c in b.context if c not in context]
    if missing_ctx:
        raise SchemaError(f"{b.type} needs {missing_ctx} (build(node, {missing_ctx[0]}=…))")
    values = _adapt(b.type, values)
    if b.varargs:
        return b.target(*args, *values.pop(b.varargs, ()), **values)
    if b.type in ("Union", "Intersection", "Difference"):
        from loudpy.geometry.selectors import as_selector
        return b.target(as_selector(values["a"]), as_selector(values["b"]))
    if b.solve is not None:
        solve = {k: v for k, v in values.items() if b.param(k).stage == "solve"}
        init = {k: v for k, v in values.items() if k not in solve}
        return b.target(*args, **init), solve
    return b.target(*args, **values)


def _adapt(type_: str, values: dict) -> dict:
    """What plain data cannot say the way the code wants it: a Force signal (samples, fs) from a signals node, a
    mapping of parameters as lp.Parameters."""
    if type_ == "Force" and isinstance(values.get("amplitude"), tuple) and len(values["amplitude"]) == 2:
        samples, fs = values["amplitude"]
        values = {**values, "amplitude": samples, "fs": values.get("fs") or fs}
    if type_ in ("Model", "Scene") and isinstance(values.get("parameters"), dict):
        from loudpy.model.parameters import Parameters
        values = {**values, "parameters": Parameters(**values["parameters"])}
    return values


def code(v, p: Param | None = None, path: Callable[[str], str] | None = None) -> str:
    """Python code for a node or a value (a script's text): lp.Solid(name='cone', material='Paper', …); path(text):
    the code of a file path (default: its repr); an Item is model['name']."""
    if isinstance(v, dict) and "type" in v:
        b = block(v["type"])
        given = v.get("params") or {}
        if b.type == "Item":
            return f"model[{given['name']!r}]"
        if b.type == "CircuitRef":                                    # the script's variable of that circuit
            return str(given["name"])
        if b.type in ("Union", "Intersection", "Difference"):
            op = {"Union": "|", "Intersection": "&", "Difference": "-"}[b.type]
            return f"({code(given['a'], b.param('a'), path)} {op} {code(given['b'], b.param('b'), path)})"
        args = [code(x, b.param(b.varargs), path) for x in given.get(b.varargs, [])] if b.varargs else []
        args += [f"{k}={code(x, b.param(k), path)}" for k, x in given.items() if k != b.varargs]
        return f"lp.{b.type}({', '.join(args)})"
    if isinstance(v, dict) and "P" in v:
        out = f"lp.P({v['P']!r})"
        if float(v.get("scale", 1.0)) != 1.0:
            out += f" * {float(v['scale'])!r}"
        if float(v.get("offset", 0.0)) != 0.0:
            out += f" + {float(v['offset'])!r}"
        return out
    if isinstance(v, dict) and "complex" in v:
        return repr(complex(*v["complex"]))
    if isinstance(v, dict) and p is not None and p.kind == "frequencies":
        fn = "np.geomspace" if v.get("spacing", "log") == "log" else "np.linspace"
        return f"{fn}({float(v['from'])!r}, {float(v['to'])!r}, {int(v.get('points', 100))})"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {code(x, None, path)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        inner = [code(x, Param("", "pair") if p is not None and p.kind == "pairs" else None, path) for x in v]
        tuple_ = p is not None and (p.kind in _TUPLE_KINDS or p.kind == "pair" or isinstance(p.default, tuple))
        if tuple_:
            return "(" + ", ".join(inner) + ("," if len(inner) == 1 else "") + ")"
        return "[" + ", ".join(inner) + "]"
    if isinstance(v, str) and p is not None and p.kind == "file" and path is not None:
        return path(v)
    return repr(v)


def check(node: dict) -> list[str]:
    """The problems of a node's values against its description (kinds, choices, limits, required parameters, nested
    nodes), before building it; [] when none. The block's own checks still run when it is built."""
    try:
        b = block(node.get("type", "")) if isinstance(node, dict) else None
    except SchemaError as e:
        return [str(e)]
    if b is None:
        return [f"a node is {{'type': …, 'params': {{…}}}}, got {node!r}"]
    out = []
    given = node.get("params") or {}
    for name in given:
        if name not in {p.name for p in b.params} and not (b.type == "Item" and name == "name"):
            out.append(f"{b.type}: unknown parameter {name!r}")
    for p in b.params:
        if p.name not in given:
            if p.required:
                out.append(f"{b.type}.{p.name}: required")
            continue
        out += [f"{b.type}.{p.name}: {e}" for e in _check_value(given[p.name], p)]
    return out


def _check_value(v, p: Param) -> list[str]:
    if v is None:
        return [] if p.optional or p.default is None else ["a value is needed"]
    if isinstance(v, dict) and "P" in v:
        return [] if p.kind in ("number", "integer", "complex", "pair", "point", "range") else \
            ["a parameter reference only replaces a number"]
    if p.kind == "frequencies" and isinstance(v, dict):
        try:
            frequencies(v)
            return []
        except (SchemaError, KeyError, TypeError, ValueError) as e:
            return [str(e) if isinstance(e, SchemaError) else f"frequencies: {{'from', 'to', 'points'}}, got {v!r}"]
    if isinstance(v, dict) and "type" in v:
        problems = check(v)
        if p.kind in ("block", "blocks") and p.of and block(v["type"]).section != p.of and not problems:
            problems.append(f"a {block(v['type']).section} block where a {p.of} block goes")
        return problems
    k = p.kind
    if k in ("number", "integer"):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (k == "integer" and int(v) != v):
            return [f"{KINDS[k]}, got {v!r}"]
        if p.min is not None and (v < p.min or (v == p.min and not p.min_inclusive)):
            return [f"{'≥' if p.min_inclusive else '>'} {p.min:g}, got {v!r}"]
        if p.max is not None and (v > p.max or (v == p.max and not p.max_inclusive)):
            return [f"{'≤' if p.max_inclusive else '<'} {p.max:g}, got {v!r}"]
        return []
    if k == "complex":
        ok = (isinstance(v, (int, float)) and not isinstance(v, bool)) or (isinstance(v, dict) and "complex" in v) \
            or isinstance(v, list)                                                   # a force signal
        return [] if ok else [f"{KINDS[k]}, got {v!r}"]
    if k == "choice":
        return [] if v in p.choices else [f"one of {list(p.choices)}, got {v!r}"]
    if k == "bool":
        return [] if isinstance(v, bool) else [f"true or false, got {v!r}"]
    if k in ("text", "file"):
        return [] if isinstance(v, str) else [f"{KINDS[k]}, got {v!r}"]
    if k in ("selection", "material"):
        return [] if isinstance(v, str) else [f"{KINDS[k]}, got {v!r}"]
    if k == "pair":
        return [] if isinstance(v, list) and len(v) == 2 else [f"two numbers, got {v!r}"]
    if k == "range":
        return [] if isinstance(v, (int, float)) or (isinstance(v, list) and len(v) == 2) else \
            [f"{KINDS[k]}, got {v!r}"]
    if k in ("point", "numbers", "integers", "texts", "pairs", "frequencies", "signal", "blocks"):
        if not isinstance(v, list):
            return [f"{KINDS[k]}, got {v!r}"]
        if k == "blocks":                                            # nodes (or points, where a block takes them)
            return [e for x in v if not (isinstance(x, list) and all(isinstance(c, (int, float)) for c in x))
                    for e in _check_value(x, replace(p, kind="block"))]
        return []
    if k == "mapping":
        return [] if isinstance(v, (dict, str)) else [f"{KINDS[k]}, got {v!r}"]
    if k == "block":
        return [f"a node of section {p.of!r}, got {v!r}"]
    return []


SECTION_TITLES = {
    "model": "Model", "materials": "Materials", "regions": "Regions", "boundaries": "Boundaries",
    "mesh_controls": "Mesh controls", "damping": "Damping", "drives": "Drives", "selectors": "Selectors",
    "profiles": "Wet profiles of a driver", "studies": "Studies", "study_options": "Study options",
    "signals": "Transient signals", "scene": "3D scene", "scene.cabinet": "Cabinets",
    "scene.cabinet_parts": "Cabinet parts", "scene.drivers": "Drivers", "scene.placement": "Driver placement",
    "scene.air": "Scene air and meshes", "scene.walls": "Elastic walls", "scene.panels": "Wall panels",
    "circuit.parts": "Circuit parts", "circuit.filters": "Filter stages", "circuit.coils": "Coil models"}


def _facts_rst(b: Block) -> str:
    """An item type's facts in a sentence (loudpy.model.facts)."""
    from loudpy.model.facts import PLACES, ROLES
    f = b.facts
    out = ["Selects " + ("domain groups" if f.selects == "domains" else "curves")]
    if f.medium:
        out.append(f"fills them with {f.medium}")
    if f.physics:
        out.append(f"physics: {f.physics}")
    if f.role:
        out.append(f"role: {ROLES[f.role]}")
    if f.key:
        out.append(f"key parameter: ``{f.key}``")
    text = "; ".join(out) + "."
    if f.fits:
        text += " Fits " + "; ".join(f"{PLACES[x.place]}" + (f" ({x.shape})" if x.shape else "") +
                                     ("" if x.usual else ", possible") for x in f.fits) + "."
    return text


def reference_rst() -> str:
    """The building blocks as a reStructuredText page (docs/guide/building_blocks.rst is this text; a test checks it
    is up to date)."""
    def cell(x) -> str:
        return str(x).replace("\n", " ").replace("|", "\\|")

    out = ["Building blocks", "===============", "",
           ".. This page is generated by loudpy.schema.reference_rst(): do not edit it by hand.", "",
           "Every block a simulation is made of, with its parameters as ``lp.schema`` describes them for programs (the",
           "interface draws its forms from this description). *value*: the kind of value (``lp.schema.KINDS``); a",
           "*block* parameter takes a block of the section named after it. Defaults come from the code.", ""]
    for sec in SECTIONS:
        bs = blocks(sec)
        if not bs:
            continue
        title = SECTION_TITLES.get(sec, sec)
        out += [title, "-" * len(title), ""]
        for b in bs:
            out += [f"``{'lp.' + b.type}``", "^" * (len(b.type) + 7), "", cell(b.summary) or "(no summary)", ""]
            if b.context:
                out += [f"Given by the context: {', '.join(b.context)}.", ""]
            if b.facts is not None and b.facts.table:
                out += [_facts_rst(b), ""]
            if b.params:
                out += [".. list-table::", "   :header-rows: 1", "   :widths: 18 14 8 16 44", "",
                        "   * - parameter", "     - value", "     - unit", "     - default", "     - meaning"]
                for p in b.params:
                    kind = p.kind + (f" ({p.of})" if p.of else "") + (", in .solve()" if p.stage == "solve" else "")
                    default = "required" if p.required else f"``{p.default!r}``"
                    extra = f" Choices: {', '.join(str(c) for c in p.choices)}." if p.choices and \
                        len(p.choices) > 1 else ""
                    if p.limits():
                        extra += f" Range: {p.limits()}."
                    out += [f"   * - ``{p.name}``", f"     - {cell(kind)}", f"     - {cell(p.unit)}",
                            f"     - {cell(default)}", f"     - {cell(p.help)}{cell(extra)}"]
                out.append("")
            if b.code_only:
                out += ["Only in a script: " + "; ".join(f"{k}: {cell(v)}" for k, v in b.code_only.items()) + ".",
                        ""]
    out += ["Not building blocks", "-------------------", "",
            "Public names of loudpy that a simulation is not made of:", ""]
    out += [f"- ``{k}``: {cell(v)}" for k, v in NOT_BLOCKS.items()]
    return "\n".join(out) + "\n"


__all__ = ["Block", "Param", "SchemaError", "KINDS", "SECTIONS", "NOT_BLOCKS", "catalog", "block", "blocks",
           "defaults", "describe", "build", "check", "code", "frequencies", "to_value", "from_value",
           "reference_rst", "register", "unregister", "item_params"]
