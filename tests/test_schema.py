"""lp.schema: every public building block described, every parameter of the code described (and no other), the
description round-trips through plain data, refuses wrong values, and the generated reference page is current."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import loudpy as lp
from loudpy import schema
from support import current_api as api

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "guide" / "building_blocks.rst"
STEP = str(Path(__file__).resolve().parent / "data" / "geometries" / "Loudspeaker" / "HPNEW-Sketch.step")


def test_every_public_name_is_a_block_or_said_not_to_be():
    targets = {id(b.target) for b in schema.catalog().values()}
    for name in lp.__all__:
        obj = getattr(lp, name)
        assert id(obj) in targets or name in schema.NOT_BLOCKS, f"lp.{name}: describe it in lp.schema or say why not"
    for name in schema.NOT_BLOCKS:
        assert name in lp.__all__, f"NOT_BLOCKS names lp.{name}, which is not public"
    for name in lp.cabinets.__all__:
        assert f"cabinets.{name}" in schema.catalog() or name == "volume"
    for name in ("sine", "multitone", "wav"):
        assert f"signals.{name}" in schema.catalog()


def _code_params(b):
    """The parameters of the code behind a block, minus self, its context and **kwargs."""
    fns = [b.target if not inspect.isclass(b.target) or b.type != "Model" else lp.Model.__init__]
    if b.solve is not None:
        fns.append(b.solve)
    out = set()
    for fn in fns:
        for n, par in inspect.signature(fn).parameters.items():
            if par.kind is inspect.Parameter.VAR_KEYWORD or n == "self":
                continue
            if par.kind is inspect.Parameter.VAR_POSITIONAL and n != b.varargs:
                continue
            out.add(n)
    return out - set(b.context)


@pytest.mark.parametrize("type_", list(schema.catalog()))
def test_every_parameter_of_the_code_is_described(type_):
    b = schema.block(type_)
    described = {p.name for p in b.params}
    hidden = {k.split(" ")[0] for k in b.code_only}
    missing = _code_params(b) - described - hidden
    assert not missing, f"{type_}: parameters of the code not described: {sorted(missing)}"
    for p in b.params:
        assert p.kind in schema.KINDS and p.help, f"{type_}.{p.name}: a kind of KINDS and a help text"
        assert not p.choices or p.kind == "choice"
        assert p.kind not in ("block", "blocks") or p.of in schema.SECTIONS, f"{type_}.{p.name}: of a section"
        if p.kind == "choice" and not p.required:
            assert p.default in p.choices, f"{type_}.{p.name}: default {p.default!r} not a choice"
    assert b.section in schema.SECTIONS and b.summary


SAMPLES = [
    lp.Solid("cone", "Paper", mesh_size=1e-3, damping=lp.Hysteretic(0.05), on=lp.Group("a", "b") | lp.Box(r=(0, .1))),
    lp.Solid("spider", "PhenolicCloth", damping=lp.Undamped(), E=lp.P("E_spider")),
    lp.Air("air", c=340.0), lp.PML("PML", thickness=lp.P("t") * 0.5, on=lp.Box(r=0.3) | lp.Box(z=0.5)),
    lp.Clamped("axis", dofs="r"), lp.Coil("coil"), lp.FluidStructure("wet", on=lp.Interface("cone", "air")),
    lp.Radiation("far", on=lp.BoundaryOf("air") & lp.Box(r=(0.2, 0.3))),
    lp.MeshSize(lp.Near((0.0, 0.1)) - lp.Group("x"), 5e-4),
    lp.Material("x", rho=1000.0, E=1e9, nu=0.3, rayleigh=(1.0, 1e-6), loss_factor=0.02),
    lp.Rayleigh(alpha=2.0), lp.Voltage(2.83, Re=8.0, Bl=7.0, Le=4e-3, n=0.8), lp.Force(1 + 2j),
    lp.PerWavelength(bands=[20, 1000], air_elements=8), lp.Output(fields=("u",), every=2, probes={"tip": (0.0, 0.01)}),
    lp.cabinets.Box(0.2, 0.3, 0.18, wall=0.015, braces=(lp.cabinets.Brace(0.08),),
                    ports=(lp.cabinets.Port((0.06, 0.06), 0.02, 0.06),)),
    lp.cabinets.Step(STEP, openings=(lp.cabinets.Port((0.06, 0.06), 0.02, 0.06, wall=0.0),), scale=1e-3),
    lp.Profile(lp.BoundaryOf("air_rear"), start=(0.05, 0.03)),
]


def _equal(a, b) -> bool:
    return a == b or repr(a) == repr(b)                      # selectors compare by what they select (repr)


@pytest.mark.parametrize("obj", SAMPLES, ids=lambda o: type(o).__name__)
def test_value_blocks_round_trip_through_plain_data(obj):
    node = schema.describe(obj)
    text = json.dumps(node)                                  # plain data
    assert schema.check(json.loads(text)) == []
    again = schema.build(json.loads(text))
    assert _equal(again, obj) and schema.describe(again) == node


def test_every_value_block_is_covered_by_a_sample():
    described = {schema.describe(o)["type"] for o in SAMPLES}
    nested = set()

    def walk(v):
        if isinstance(v, dict):
            if "type" in v:
                nested.add(v["type"])
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    for o in SAMPLES:
        walk(schema.describe(o))
    value_sections = {"materials", "regions", "boundaries", "mesh_controls", "damping", "drives", "selectors",
                      "profiles", "study_options", "scene.cabinet", "scene.cabinet_parts"}
    expected = {b.type for b in schema.catalog().values() if b.section in value_sections} - {"Item"}
    assert expected <= described | nested, sorted(expected - described - nested)


def test_defaults_and_refusals():
    node = schema.defaults("PML")
    assert node["params"]["thickness"] == 1.0 and node["params"]["strength"] == 3.0
    assert schema.block("Solid").param("mesh_size").label() == "mesh size [m]"
    assert schema.block("FrequencyResponse").groups()[:2] == ["Drive", "Study"]
    bad = {"type": "Solid", "params": {"name": "x", "material": "Paper", "mesh_size": -1, "dofs": "r",
                                       "damping": {"type": "Force", "params": {"amplitude": 1.0}}}}
    problems = schema.check(bad)
    assert "Solid: unknown parameter 'dofs'" in problems and "Solid.mesh_size: > 0, got -1" in problems
    assert any("drives block where a damping block goes" in p for p in problems)
    assert schema.check({"type": "Clamped", "params": {"name": "a", "dofs": "x"}}) == \
        ["Clamped.dofs: one of ['rz', 'r', 'z'], got 'x'"]
    assert schema.check({"type": "Air", "params": {}}) == ["Air.name: required"]
    assert schema.check({"type": "Nope"})[0].startswith("no building block 'Nope'")
    with pytest.raises(schema.SchemaError, match="unknown parameter"):
        schema.build({"type": "Coil", "params": {"name": "c", "dofs": "r"}})
    with pytest.raises(schema.SchemaError, match="needs \\['model'\\]"):
        schema.build({"type": "Modes", "params": {"f_max": 1000.0}})
    with pytest.raises(schema.SchemaError, match="cannot describe"):
        schema.describe(lp.Modes(api.plate_model()))
    assert schema.to_value(lp.P("h") * 2 + 1) == {"P": "h", "scale": 2.0, "offset": 1.0}


def test_studies_and_item_selections_are_built_on_their_model():
    model = api.plate_model()
    study, kw = schema.build({"type": "Modes", "params": {"f_max": 3000.0, "orders": [0, 1],
                                                          "output": {"type": "Output", "params": {"every": 2}}}},
                             model=model)
    assert isinstance(study, lp.Modes) and kw["f_max"] == 3000.0 and kw["orders"] == [0, 1] and kw["output"].every == 2
    fr, kw = schema.build({"type": "FrequencyResponse", "params": {
        "drive": {"type": "Force", "params": {"amplitude": {"complex": [1.0, 0.5]}}},
        "freqs": list(np.geomspace(20, 2000, 5)),
        "remesh": {"type": "PerWavelength", "params": {"bands": [20, 500]}}}}, model=model)
    assert fr.drive == lp.Force(1 + 0.5j) and len(kw["freqs"]) == 5 and kw["remesh"].bands == [20, 500]
    swept = lp.Model(api.PLATE_GEO, parameters=lp.Parameters(size=0.004)).add(
        schema.build({"type": "Solid", "params": {"name": "plate", "material": "Steel", "mesh_size": {"P": "size"}}}))
    sweep = schema.build({"type": "Sweep", "params": {"parameters": {"size": [0.004, 0.003]}}},
                         study=schema.build({"type": "Modes", "params": {"f_max": 3000.0}}, model=swept)[0])
    assert isinstance(sweep, lp.Sweep)
    item = model.regions[0]
    sel = schema.build({"type": "Item", "params": {"name": item.name}}, items={item.name: item})
    assert schema.describe(sel) == {"type": "Item", "params": {"name": item.name}}
    samples, fs = schema.build({"type": "signals.sine", "params": {"f": 100.0, "fs": 8000.0}})
    assert fs == 8000.0 and len(samples) > 100


def test_the_reference_page_is_current():
    assert REFERENCE.read_text() == schema.reference_rst(), \
        "docs/guide/building_blocks.rst is stale: write lp.schema.reference_rst() to it"


LIMIT_SAMPLES = {"Solid": {"name": "s", "material": "Paper"}, "Air": {"name": "a"}, "PML": {"name": "p"},
          "MeshSize": {"on": "x", "size": 0.001},
          "Material": {"name": "m", "kind": "solid", "rho": 1000.0, "E": 1e9, "nu": 0.3},
          "Voltage": {"amplitude": 1.0, "Re": 8.0, "Bl": 7.0}, "Force": {"amplitude": 1.0},
          "cabinets.Box": {"width": 0.2, "height": 0.3, "depth": 0.2}, "cabinets.Brace": {"y": 0.1},
          "cabinets.Port": {"at": [0.05, 0.05], "radius": 0.02, "length": 0.05}, "cabinets.Step": {"path": STEP},
          "signals.sine": {"f": 100.0}, "signals.multitone": {"freqs": [100.0, 200.0]},
          "Modes": {"f_max": 1000.0},
          "Transient": {"drive": {"type": "Force", "params": {"amplitude": [0.0, 1.0, 0.0], "fs": 1000.0}}},
          "Adaptive": {}, "Sweep": {"parameters": {}},
          "PerWavelength": {"bands": [20.0]}, "Output": {}, "Rayleigh": {}, "Hysteretic": {}, "Box": {}}


def _outcome(t, p, value, model):
    """"refused" when building the block (and solving, for a .solve() argument) is refused because of this value."""
    import warnings
    node = {"type": t, "params": {**LIMIT_SAMPLES[t], p.name: value}}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if t in ("Adaptive", "Sweep"):
                schema.build(node, study=lp.Modes(model))
                return "accepted"
            out = schema.build(node, model=model)
            if isinstance(out, tuple) and p.stage == "solve":
                out[0].solve(**out[1])
    except Exception as e:                                          # noqa: BLE001
        return "refused" if p.name in str(e) or "range" in str(e) else f"other: {type(e).__name__}: {e}"
    return "accepted"


def test_declared_limits_are_what_the_blocks_enforce(tmp_path):
    """Every limit of the description (min / max, inclusive or not) is the block's own: a value beyond it is refused
    by the block itself (built, or solved for a .solve() argument), an inclusive edge is accepted, an exclusive one
    refused — the forms refuse exactly what loudpy refuses."""
    model = api.plate_model()
    model.mesh(tmp_path / "plate.msh")
    checked, wrong = 0, []
    for t in LIMIT_SAMPLES:
        for p in schema.block(t).params:
            for lim, incl, sign in ((p.min, p.min_inclusive, -1), (p.max, p.max_inclusive, 1)):
                if lim is None:
                    continue
                step = 1 if p.kind == "integer" else (0.5 if lim == 0 else abs(lim) * 0.5)
                edge = int(lim) if p.kind == "integer" else float(lim)
                for value, want in ((lim + sign * step, "refused"), (edge, "accepted" if incl else "refused")):
                    got = _outcome(t, p, value, model)
                    checked += 1
                    if got != want and not (want == "accepted" and got.startswith("other")):
                        wrong.append(f"{t}.{p.name} = {value}: the description says {want}, the block {got}")
                    assert schema.check({"type": t, "params": {**LIMIT_SAMPLES[t], p.name: value}}) == [] or \
                        want == "refused" or p.stage == "solve", (t, p.name, value)
    assert not wrong, "\n".join(wrong)
    assert checked > 60
    target = schema.block("Adaptive").param("target")
    assert target.limits() == "> 0, < 1"
    assert schema.check({"type": "Adaptive", "params": {"target": 2.0}}) == ["Adaptive.target: < 1, got 2.0"]
