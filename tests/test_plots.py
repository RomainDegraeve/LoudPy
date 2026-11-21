"""Plotting (phase J): image comparisons against committed baselines, the global matplotlib settings left alone,
selections drawn on the mesh, the interactive HTML viewer.

Baselines: tests/reference/images/<name>.png, written with --update-reference. The images are compared with the
texts hidden and the axes placed on their grid cells (fonts differ between machines, and so does a
font-dependent layout) at a low resolution, by RMS difference (matplotlib.testing)."""
import base64
import json
import re
from pathlib import Path

import numpy as np
import pytest

import loudpy as lp
from support import current_api as api

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

IMAGES = Path(__file__).parent / "reference" / "images"
TOL = 6.0                                     # RMS difference (0–255 scale) accepted between machines


def _legacy(name):
    return Path(__file__).parent / "data" / "legacy" / name


def check_image(fig, name, request):
    """Compare a figure (texts hidden) with tests/reference/images/<name>.png."""
    import matplotlib.pyplot as plt
    from matplotlib.testing.compare import compare_images
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        t.set_visible(False)
    # layout independent of the fonts: no layout engine, every axes at its grid cell, colour bars hidden
    fig.set_layout_engine("none")
    for a in fig.axes:
        spec = a.get_subplotspec()
        if spec is None:
            a.set_visible(False)
        else:
            a.set_position(spec.get_position(fig))
    IMAGES.mkdir(parents=True, exist_ok=True)
    expected = IMAGES / f"{name}.png"
    out = Path(request.config.rootpath) / "tests" / ".cache" / "images"
    out.mkdir(parents=True, exist_ok=True)
    actual = out / f"{name}.png"
    fig.savefig(actual, dpi=50)
    plt.close(fig)
    if request.config.getoption("--update-reference") or not expected.exists():
        if not request.config.getoption("--update-reference"):
            pytest.fail(f"no baseline {expected.name}: run with --update-reference")
        expected.write_bytes(actual.read_bytes())
        return
    err = compare_images(str(expected), str(actual), tol=TOL)
    assert err is None, err


def _curve():
    f = np.geomspace(20, 20000, 200)
    return lp.Data(1 / (1 + 1j * f / 1000) ** 2, {"f": f}, "p")


def test_image_curve_and_map(request):
    ax = lp.plot.line([_curve(), _curve() * 0.5])
    check_image(ax.figure, "curve", request)
    ang = np.arange(0, 181, 5.0)
    f = np.geomspace(100, 10000, 60)
    d = lp.Data(np.cos(np.radians(ang))[None, :] ** 2 * np.exp(-f[:, None] / 5000) + 1e-3, {"f": f, "angle": ang}, "p")
    check_image(lp.plot.map(d).figure, "map", request)
    check_image(lp.plot.polar(d.sel(f=1000)).figure, "polar", request)


def test_image_field_and_modes(request):
    r = lp.load(_legacy("freq_meca.h5"))
    u = r.field("u", f=100.0)
    check_image(u.plot("z").figure, "field_u", request)     # (the deformed plot renders differently on Linux)
    modes = lp.load(_legacy("eigen_fourier.h5"))
    check_image(lp.plot.mode_chart(modes, y="order").figure, "modes", request)   # deformed shapes differ on Linux


def test_global_style_untouched(tmp_path, mesh_dir):
    """Every lp.plot function draws in the loudpy style and leaves the global matplotlib settings as they were."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    before = dict(mpl.rcParams)
    r = lp.load(_legacy("freq_meca.h5"))
    u = r.field("u", f=100.0)
    lp.plot.line(_curve())
    u.plot()
    lp.plot.field(u, show="z")
    lp.plot.boundary(u, r.boundary("interface_acou_meca_front"))
    lp.plot.modes(lp.load(_legacy("eigen_fourier.h5")).order(0), indices=[0])
    fig, ax = lp.plot.figure("wide")
    lp.plot.selection(api.plate_problem(mesh_dir), lp.Box(r=0))
    lp.plot.html(u, tmp_path / "u.html")
    lp.plot.save(fig, tmp_path / "x.pdf")
    plt.close("all")
    after = dict(mpl.rcParams)
    changed = {k for k in before if before[k] != after[k] and k != "backend"}
    assert not changed, changed
    assert not lp.plot.style.is_active()


def test_selection_2d(mesh_dir, request):
    model = api.plate_problem(mesh_dir)
    ax = lp.plot.selection(model, lp.Box(r=0))
    from matplotlib.collections import LineCollection
    lines = [c for c in ax.collections if isinstance(c, LineCollection) and len(c.get_segments()) and
             np.allclose(np.asarray(c.get_segments()[0])[:, 0], 0.0)]
    assert lines and len(lines[0].get_segments()) == len(model.select(lp.Box(r=0)))
    domains = model.select(model.regions[0], kind="domain")
    ax2 = lp.plot.selection(model, model.regions[0], kind="domain")
    assert domains and "domain group" in ax2.get_title()
    check_image(ax.figure, "selection_2d", request)
    with pytest.raises(lp.errors.SelectionError):
        lp.plot.selection(model, lp.Box(r=123.0))


def test_selection_3d_surfaces():
    from loudpy.geometry.mesh import Mesh, MeshGroup
    from support.fem_meshes import box_cavity
    nodes, tets, faces = box_cavity(0.3, 0.2, 0.1, 0.08)
    groups = {"air": MeshGroup("air", 3, "tet10", tets)}
    groups.update({k: MeshGroup(k, 2, "tri6", v) for k, v in faces.items()})
    mesh = Mesh(np.arange(len(nodes)), nodes, groups)
    ax = lp.plot.selection(mesh, ["z1", "x0"])
    assert ax.name == "3d" and len(ax.collections) == 6
    with pytest.raises(lp.errors.SelectionError, match="no surface"):
        lp.plot.selection(mesh, "walls")


def _viewer_data(path):
    text = Path(path).read_text()
    data = json.loads(re.search(r"const D=(\{.*?\});\n", text, re.S).group(1))
    dec = lambda s, t: np.frombuffer(base64.b64decode(s), t)
    return text, data, dec


def test_html_viewer(tmp_path):
    """One self-contained file: no external script, the surface and the values decode, frames share the range."""
    r = lp.load(_legacy("freq_meca.h5"))
    u = r.field("u", f=100.0)
    path = u.export(tmp_path / "u.html")
    text, data, dec = _viewer_data(path)
    assert "<script src" not in text and "http" not in text.split("<script>")[1]
    pos, tri = dec(data["pos"], np.float32).reshape(-1, 3), dec(data["tri"], np.uint32).reshape(-1, 3)
    assert len(pos) == data["n"] and len(tri) == data["n_tri"] and tri.max() < data["n"]
    vals = dec(data["frames"][0], np.float32)
    assert len(vals) == data["n"] and data["lo"] <= vals.min() + 1e-6 and vals.max() <= data["hi"] + 1e-6
    np.testing.assert_allclose(vals.max(), np.abs(u.values).max(axis=-1).max() if u.is_vector else
                               np.abs(u.values).max(), rtol=1e-5)
    # frames (mirrored section): one value array per frame, the triangles doubled
    path2 = lp.plot.html([u, r.field("u", f=100.0)], tmp_path / "u2.html", show="z", mirror=True)
    _, d2, _ = _viewer_data(path2)
    assert len(d2["frames"]) == 2 and d2["n_tri"] == 2 * data["n_tri"] and d2["hi"] > d2["lo"]


def test_html_viewer_3d(tmp_path):
    """A 3D field: the outer surface of the tetrahedra (closed box: every face once)."""
    from loudpy.geometry.mesh import Mesh, MeshGroup
    from support.fem_meshes import box_cavity
    nodes, tets, faces = box_cavity(0.3, 0.2, 0.1, 0.08)
    mesh = Mesh(np.arange(len(nodes)), nodes, {"air": MeshGroup("air", 3, "tet10", tets)})
    p = lp.Field("p", mesh, np.cos(np.pi * nodes[:, 0] / 0.3) + 0j, 500.0, "f")
    _, data, dec = _viewer_data(lp.plot.html(p, tmp_path / "p.html", show="real"))
    assert data["n_tri"] == sum(len(v) for v in faces.values())


def _can_render() -> bool:
    """pyvista renders off screen here (checked in a subprocess: a missing OpenGL can abort the process)."""
    import subprocess
    import sys
    code = ("import pyvista as pv; pl = pv.Plotter(off_screen=True); pl.add_mesh(pv.Sphere()); "
            "pl.screenshot(return_img=True)")
    try:
        return subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120).returncode == 0
    except Exception:
        return False


def test_view3d_saves(tmp_path):
    """A 3D view of a 2D displacement (revolved) saved as a self-contained HTML viewer (the colours of each
    dataset baked per vertex) and as a ParaView multiblock with its values; a 2D pressure has no 3D view."""
    pv = pytest.importorskip("pyvista")
    u = lp.load(_legacy("freq_meca.h5")).field("u", f=100.0)
    v = u.view3d(sweep=np.pi)
    text, data, dec = _viewer_data(v.save(tmp_path / "u.html", title="u at 100 Hz"))
    assert "<script src" not in text and data["title"] == "u at 100 Hz" and not data["nobar"]
    tri, rgb = dec(data["tri"], np.uint32).reshape(-1, 3), dec(data["rgb"], np.uint8).reshape(-1, 3)
    surf = v.items[0][0].triangulate()
    assert len(rgb) == data["n"] == surf.n_points and tri.max() < data["n"] and data["n_tri"] == surf.n_cells
    assert rgb.std() > 10                                            # coloured by |u|, not uniform
    blocks = pv.read(v.save(tmp_path / "u.vtm"))
    assert blocks.n_blocks == 1 and "|u|" in blocks[0].point_data
    np.testing.assert_allclose(blocks[0]["|u|"], v.items[0][0]["|u|"], rtol=1e-6)
    from loudpy.errors import ResultsError
    p2d = lp.Field("p", u.mesh, np.ones(u.mesh.n_nodes, complex), 100.0, "f")
    with pytest.raises(ResultsError, match="revolves a displacement"):
        p2d.view3d()
    with pytest.raises(ValueError, match="unknown extension"):
        v.save(tmp_path / "u.xyz")


def test_view3d_cut_and_render(tmp_path):
    """A 3D field cut by a plane: only the kept side is drawn; rendered images (PNG figure, glTF binary model)."""
    pytest.importorskip("pyvista")
    from loudpy.geometry.mesh import Mesh, MeshGroup
    from support.fem_meshes import box_cavity
    nodes, tets, _ = box_cavity(0.3, 0.2, 0.1, 0.08)
    mesh = Mesh(np.arange(len(nodes)), nodes, {"air": MeshGroup("air", 3, "tet10", tets)})
    p = lp.Field("p", mesh, np.cos(np.pi * nodes[:, 0] / 0.3) + 0j, 500.0, "f")
    v = lp.plot.view(p, show="real", cut=((1, 0, 0), (0.15, 0, 0)))
    pts = v.items[0][0].points
    assert pts[:, 0].max() <= 0.15 + 1e-9 or pts[:, 0].min() >= 0.15 - 1e-9          # one side of the plane
    assert v.bars[0][1][0] == pytest.approx(-v.bars[0][1][1])                         # symmetric range (real part)
    if not _can_render():
        pytest.skip("no off-screen OpenGL here")
    png = v.save(tmp_path / "p.png", scale=1, dpi=100)
    assert png.stat().st_size > 10_000
    glb = v.save(tmp_path / "p.glb").read_bytes()
    import struct
    magic, version, length = struct.unpack("<III", glb[:12])
    assert magic == 0x46546C67 and version == 2 and length == len(glb)
    n_json = struct.unpack("<I", glb[12:16])[0]
    doc = json.loads(glb[20:20 + n_json])
    assert doc["meshes"] and doc["buffers"][0]["byteLength"] == len(glb) - 20 - n_json - 8
