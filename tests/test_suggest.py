"""
Item types declare their facts (loudpy.model.facts) next to their classes; lp.schema shows them on each Block (the key
parameter marked); lp.suggest reads the geometry only (sides, axis, shape) to say which types fit each group;
lp.readiness reads the declared roles. A new item type registered once (lp.schema.register) takes part everywhere.
"""
from dataclasses import dataclass

import pytest

import loudpy as lp
from loudpy.model.boundaries import Boundary
from loudpy.model.facts import Facts, Fit, facts_of, item_types
from support import current_api as api


def test_every_item_type_declares_its_facts_and_schema_shows_them():
    types = item_types()
    assert list(types)[:3] == ["Solid", "Air", "PML"] and "Clamped" in types and "MeshSize" in types
    for name, cls in types.items():
        b = lp.schema.block(name)
        assert b.facts is cls.facts and b.facts is facts_of(name)
        if cls.facts.key:
            assert b.key is not None and b.key.name == cls.facts.key and b.key.key
    assert lp.schema.block("Solid").key.name == "material" and lp.schema.block("FluidStructure").key is None
    assert facts_of(lp.Air("air")).medium == "air" and facts_of(lp.PML).selects == "curves"
    assert {facts_of(t).role for t in ("Clamped", "Coil", "FluidStructure", "Radiation", "PML")} == \
        {"held", "drive", "coupling", "absorbing"}
    assert lp.schema.block("Modes").facts is None
    with pytest.raises(ValueError):
        Fit("nowhere", "")


def _dtn_model():
    m = lp.Model(api.DTN_GEO, api.MATERIALS)
    m.add([lp.Solid(n, mat) for n, mat in api.SPEAKER_PARTS], lp.Air("subacou"))
    return m


def test_suggest_reads_the_geometry_only():
    """Places from the media on each side of the curves, the axis and the curve shapes (the test knows the group
    names of its geometry; lp.suggest never reads them)."""
    m = _dtn_model()
    s = {x.group: x for x in lp.suggest(m)}
    assert s["submeca_coil"].kind == "domain" and s["submeca_coil"].types == ["Solid", "Air"]
    assert s["submeca_coil"].items == ["coil"]
    assert (s["interface_acou_meca_front"].place, s["interface_acou_meca_front"].best) == ("solid|air",
                                                                                         "FluidStructure")
    assert (s["interface_constrained_r_1"].place, s["interface_constrained_r_1"].best) == ("solid|axis", "Clamped")
    assert s["interface_constrained_r_1"].params == {"Clamped": {"dofs": "r"}}
    assert (s["interface_constrained_rz_2"].place, s["interface_constrained_rz_2"].best) == ("solid|-", "Clamped")
    forced = s["interface_forced_z_1"]
    assert forced.place == "solid|solid" and forced.best is None and forced.types == ["Coil"]
    assert "two solid parts" in forced.why
    arc = s["dtn_boundary_front"]                                  # an arc of the air: exact radiation, no PML
    assert (arc.place, arc.shape, arc.usual) == ("air|-", "arc", ["Radiation"])
    pml = {x.group: x for x in lp.suggest(api.speaker_model(size=4e-3, acoustic=True))}["PML_z+_1_front"]
    assert (pml.place, pml.shape, pml.usual, pml.items) == ("air|-", "straight", ["PML"], ["PML"])
    bare = {x.group: x for x in lp.suggest(lp.Model(api.DTN_GEO))}  # no regions: no media, no place
    assert bare["interface_acou_meca_front"].place == "" and bare["interface_acou_meca_front"].types == []
    assert lp.suggest(object()) == []                              # (a scene: no geometry groups)


def test_readiness_reads_the_roles():
    m = _dtn_model()
    steps = {s.key: s for s in lp.readiness(m)}
    assert steps["held"].state == "warning" and "Clamped" in steps["held"].message
    assert steps["coupling"].state == "warning" and "FluidStructure" in steps["coupling"].message
    assert steps["radiation"].state == "warning" and "a PML or a Radiation" in steps["radiation"].message
    m.add(lp.FluidStructure("wet", on=lp.Group("interface_acou_meca_front")), lp.Radiation("dtn_boundary"),
          lp.Clamped("interface_constrained_rz"))
    steps = {s.key: s for s in lp.readiness(m)}
    assert steps["held"].state == "done" and steps["radiation"].state == "done"
    assert steps["coupling"].state == "warning" and steps["coupling"].items == ["interface_acou_meca_rear"]
    m.boundaries[0] = lp.FluidStructure("wet", on=lp.Group("interface_acou_meca_front", "interface_acou_meca_rear"))
    assert {s.key: s for s in lp.readiness(m)}["coupling"].state == "done"


@dataclass
class Absorber(Boundary):
    """A test item type: an impedance-like end of the air (only its facts matter here)."""
    strength: float = 1.0
    facts = Facts(selects="curves", physics="air", role="absorbing", key="strength",
                  fits=(Fit("air|-", "an outer edge of the air: absorbed there"),))


def test_a_new_item_type_takes_part_everywhere():
    b = lp.schema.register(Absorber, [lp.schema.Param("strength", "number", help="how much", group="Layer", min=0)])
    try:
        assert b.type == "Absorber" and b.section == "boundaries" and b.key.name == "strength"
        assert [p.name for p in b.params] == ["name", "on", "strength"] and b.param("strength").default == 1.0
        assert "Absorber" in [x.type for x in lp.schema.blocks("boundaries")] and "Absorber" in item_types()
        m = api.speaker_model(size=4e-3, acoustic=True)
        m.regions[:] = [r for r in m.regions if facts_of(r).selects == "domains"]          # no PML
        s = {x.group: x for x in lp.suggest(m)}["PML_z+_1_front"]
        assert "Absorber" in s.usual and "PML" in s.usual
        steps = {x.key: x for x in lp.readiness(m)}
        assert steps["radiation"].state == "warning" and "an Absorber" in steps["radiation"].message
        node = lp.schema.defaults("Absorber")
        node["params"].update(name="end", on="PML_z+_1_front")
        m.add(lp.schema.build(node))
        assert {x.key: x for x in lp.readiness(m)}["radiation"].state == "done"
        assert lp.schema.describe(m["end"]) == {"type": "Absorber", "params": {"name": "end", "on": "PML_z+_1_front"}}
    finally:
        lp.schema.unregister("Absorber")
    assert "Absorber" not in item_types() and "Absorber" not in lp.schema.catalog()
