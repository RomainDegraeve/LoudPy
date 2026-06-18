from __future__ import annotations
import h5py
import numpy as np
from pathlib import Path

from loudpy.Files_Loader.Load_Domains import Domain
from loudpy.Files_Loader.Core_Models import Mesh, FreqSnapshot, ModeSnapshot, TimeSnapshot, TimeRun
from loudpy.Files_Loader.Mesh_Loader import MeshLoader


def _field_to_domain(field: str) -> Domain:
    """Map a field name to its physics domain."""
    return Domain.ACOU if "acou" in field else Domain.MECA


class BaseReader:
    """Base HDF5 reader (context-managed)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._h5: h5py.File | None = None

    def __enter__(self):
        self._h5 = h5py.File(self.path, "r")
        return self

    def __exit__(self, *exc):
        self._h5.close()
        self._h5 = None

    # ---- shared services -----------------------------------------------------
    def _first_mesh_id(self) -> str:
        return next(iter(self._h5["meshes"].keys()))

    def mesh(self, mesh_id: str, domain: Domain | str) -> Mesh:
        return MeshLoader.load(self._h5, mesh_id, domain)

    def subdomain_names(self, mesh_id: str | None = None) -> list[str]:
        """List named subdomains stored for this mesh."""
        mesh_id = mesh_id or self._first_mesh_id()
        path = f"meshes/{mesh_id}/subdomains"
        if path not in self._h5:
            return []
        return list(self._h5[path].keys())

    def subdomain_mesh(self, name: str, mesh_id: str | None = None) -> Mesh:
        """Load the Mesh for a single named subdomain (e.g. 'cone')."""
        mesh_id = mesh_id or self._first_mesh_id()
        g    = self._h5[f"meshes/{mesh_id}/subdomains/{name}"]
        tags = g["tags"][:]
        crds = g["coords"][:]
        tris = g["tris"][:]                        # local row indices (0..n_nodes-1)
        return Mesh(tags, crds, tris)

    def interface_names(self) -> list[str]:
        """List named interfaces stored in the file."""
        if "interfaces" not in self._h5:
            return []
        return list(self._h5["interfaces"].keys())

    @property
    def interfaces(self) -> dict:
        if "interfaces" not in self._h5:
            return {}
        return {
            name: {
                "kind":      g.attrs.get("kind", ""),
                "node_tags": g["node_tags"][:] if "node_tags" in g else np.array([]),
                "attrs":     dict(g.attrs),
                "datasets":  {k: g[k] for k in g.keys()},
            }
            for name, g in self._h5["interfaces"].items()
        }

    def extract_subdomain(self, snap, subdomain: str, field: str,
                          mesh_id: str | None = None) -> tuple:
        """
        Extract field values restricted to a named subdomain.

        Returns
        -------
        sub_mesh : Mesh
            Mesh of the subdomain only.
        values : np.ndarray
            Field values at subdomain nodes, shape (n_sub_nodes,) or
            (n_sub_nodes, dpn). Index i aligns with sub_mesh.coords[i].
        """
        mesh_id    = mesh_id or self._first_mesh_id()
        sub_mesh   = self.subdomain_mesh(subdomain, mesh_id)
        global_field = snap.fields[field]          # (n_global_nodes[, dpn])

        # global mesh needed for tag→row lookup
        global_mesh = self.mesh(mesh_id, _field_to_domain(field))
        t2r = global_mesh.tag_to_row
        rows = np.array([t2r[int(t)] for t in sub_mesh.tags if int(t) in t2r])
        valid = np.array([int(t) in t2r for t in sub_mesh.tags])

        values = global_field[rows] if valid.all() else global_field[rows[valid]]
        return sub_mesh, values

    def extract_interface(self, snap, 
                          interface: str, field: str,mesh_id: str | None = None) -> tuple:
        """
        Extract field values along a named interface.

        Returns
        -------
        coords : np.ndarray  (n_iface_nodes, 2)
        values : np.ndarray  (n_iface_nodes,) or (n_iface_nodes, dpn)
        """
        mesh_id      = mesh_id or self._first_mesh_id()
        iface_tags   = self._h5[f"interfaces/{interface}/node_tags"][:]
        global_mesh  = self.mesh(mesh_id, _field_to_domain(field))
        t2r          = global_mesh.tag_to_row
        rows         = np.array([t2r[int(t)] for t in iface_tags if int(t) in t2r])
        coords       = global_mesh.coords[rows]
        values       = snap.fields[field][rows]
        return coords, values

    def dump(self) -> None:
        def walk(node, ind=0):
            pad = "  " * ind
            for k, v in node.attrs.items():
                print(f"{pad}  @{k} = {v!r}")
            if isinstance(node, h5py.Dataset):
                print(f"{pad}  shape={node.shape}  dtype={node.dtype}")
                return
            for name, child in node.items():
                tag = "DS" if isinstance(child, h5py.Dataset) else "GR"
                print(f"{pad}[{tag}] {name}")
                walk(child, ind + 1)
        print(f"FILE: {self.path}\n"); walk(self._h5)

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _decode_complex(arr: np.ndarray) -> np.ndarray:
        if arr.dtype.names and {"r", "i"} <= set(arr.dtype.names):
            return arr["r"] + 1j * arr["i"]
        return arr

    @staticmethod
    def _stats(arr: np.ndarray) -> str:
        """Return a compact statistics summary for any numeric array."""
        try:
            if np.iscomplexobj(arr):
                mag = np.abs(arr)
                return (f"|min|={mag.min():.4g}, |max|={mag.max():.4g}, "
                        f"|mean|={mag.mean():.4g}  (complex)")
            a = np.asarray(arr)
            if a.size == 0:
                return "(empty)"
            return (f"min={a.min():.4g}, max={a.max():.4g}, "
                    f"mean={a.mean():.4g}, std={a.std():.4g}")
        except Exception:
            return f"(stats N/A, dtype={arr.dtype})"

    @staticmethod
    def _fmt_attrs(attrs, indent: str = "║       ") -> None:
        """Pretty-print all HDF5 attributes."""
        if not len(attrs):
            print(f"{indent}(no attrs)")
            return
        for ak, av in attrs.items():
            if isinstance(av, (bytes, np.bytes_)):
                av = av.decode("utf-8", errors="replace")
            if isinstance(av, np.ndarray) and av.size > 6:
                print(f"{indent}@{ak} = ndarray shape={av.shape} dtype={av.dtype}")
            else:
                print(f"{indent}@{ak} = {av!r}")

    # ---- mesh / interface printers ------------------------------------------
    def _print_meshes(self) -> None:
        if "meshes" not in self._h5:
            print("╠══ meshes: (none)")
            return
        meshes = self._h5["meshes"]
        print(f"╠══ meshes ({len(meshes)}):")
        for mid, mg in meshes.items():
            print(f"║  ── mesh_id = '{mid}'")
            n_nodes = mg.attrs.get("n_nodes", "?")
            n_tris  = mg.attrs.get("n_tris",  "?")
            print(f"║      n_nodes={n_nodes}  n_tris={n_tris}")
            # named subdomains
            if "subdomains" in mg:
                names = list(mg["subdomains"].keys())
                print(f"║      subdomains ({len(names)}): {names}")
                for sname, sg in mg["subdomains"].items():
                    phys = sg.attrs.get("physics", "?")
                    nn   = len(sg["tags"]) if "tags" in sg else "?"
                    print(f"║        • '{sname}'  physics={phys}  n_nodes={nn}")
            for sub_name, sub in mg.items():
                if sub_name == "subdomains":
                    continue
                if isinstance(sub, h5py.Group):
                    print(f"║      ▸ physics group '{sub_name}':")
                    for dname, dset in sub.items():
                        if isinstance(dset, h5py.Dataset):
                            print(f"║          • {dname}: shape={dset.shape}")
                elif isinstance(sub, h5py.Dataset):
                    print(f"║      • {sub_name}: shape={sub.shape}, "
                          f"dtype={sub.dtype}")

    def _print_interfaces(self) -> None:
        ifs = self.interfaces
        if not ifs:
            return
        print(f"╠══ interfaces ({len(ifs)}):")
        for n, d in ifs.items():
            print(f"║  • {n}")
            print(f"║      kind     : {d['kind']}")
            print(f"║      n_nodes  : {len(d['node_tags'])}")
            if len(d['node_tags']):
                tags = d['node_tags']
                print(f"║      tag range: [{int(tags.min())}, {int(tags.max())}]")
            if d['attrs']:
                print(f"║      attrs    :")
                self._fmt_attrs(d['attrs'], indent="║         ")
            for ds_name, dset in d['datasets'].items():
                if isinstance(dset, h5py.Dataset):
                    print(f"║      • dataset '{ds_name}': "
                          f"shape={dset.shape}, dtype={dset.dtype}")

    def _print_root_attrs(self) -> None:
        if len(self._h5.attrs):
            print(f"╠══ root attrs:")
            self._fmt_attrs(self._h5.attrs, indent="║    ")

    def _print_top_level_groups(self) -> None:
        known = {"meshes", "interfaces", "fields", "dof_maps",
                 "modes", "time_histories"}
        extras = [k for k in self._h5.keys() if k not in known]
        if extras:
            print(f"╠══ other top-level groups: {extras}")

    # ---- main entry ----------------------------------------------------------
    def inspect(self) -> None:
        """Print a verbose human-readable summary of the HDF5 file."""
        print(f"╔══ {type(self).__name__}: {self.path.name}")
        print(f"║   path: {self.path}")
        try:
            size_mb = self.path.stat().st_size / 1e6
            print(f"║   size: {size_mb:.3f} MB")
        except Exception:
            pass
        self._print_root_attrs()
        self._print_meshes()
        self._print_interfaces()
        self._print_top_level_groups()
        self._inspect_specific()
        print("╚══ tip: call `.dump()` for the full HDF5 tree.\n")

    def _inspect_specific(self) -> None:
        """Override in subclasses."""


class FreqReader(BaseReader):
    """One file → one or more harmonic snapshots."""

    def snapshots(self) -> list[FreqSnapshot]:
        g = self._h5["fields/freq"]
        out: list[FreqSnapshot] = []
        for sid in sorted(g.keys()):
            snap = g[sid]
            fields   = {n: self._decode_complex(snap[n][:]) for n in snap}
            dm_g     = self._h5.get(f"dof_maps/{sid}", {})
            dof_maps = {k: dm_g[k][:] for k in dm_g}
            out.append(FreqSnapshot(
                label    = f"f={float(snap.attrs['value']):.2f}Hz",
                fields   = fields,
                dof_maps = dof_maps,
                mesh_id  = snap.attrs["mesh_id"],
                meta     = dict(snap.attrs),
                f        = float(snap.attrs["value"]),
            ))
        return out

    def load(self) -> FreqSnapshot:
        snaps = self.snapshots()
        if len(snaps) != 1:
            raise RuntimeError(f"Expected 1 snapshot, got {len(snaps)}")
        return snaps[0]

    def load_with_meshes(self) -> tuple[FreqSnapshot, Mesh, Mesh]:
        s = self.load()
        return s, self.mesh(s.mesh_id, Domain.ACOU), self.mesh(s.mesh_id, Domain.MECA)

    def _inspect_specific(self) -> None:
        if "fields/freq" not in self._h5:
            print("╠══ fields/freq: (none)")
            return
        g = self._h5["fields/freq"]
        sids = sorted(g.keys())
        print(f"╠══ frequency snapshots ({len(sids)}):")
        for sid in sids:
            snap = g[sid]
            f_val = float(snap.attrs.get('value', float('nan')))
            print(f"║")
            print(f"║  ── snapshot '{sid}' @ f = {f_val:.4f} Hz")
            print(f"║      mesh_id : {snap.attrs.get('mesh_id', '?')}")
            print(f"║      attrs   :")
            self._fmt_attrs(snap.attrs, indent="║         ")
            print(f"║      fields  :")
            for fname in snap.keys():
                dset = snap[fname]
                arr  = self._decode_complex(dset[:])
                print(f"║         • {fname}: shape={dset.shape}, "
                      f"dtype={dset.dtype}")
                print(f"║              stats: {self._stats(arr)}")
            dm_path = f"dof_maps/{sid}"
            if dm_path in self._h5:
                dm = self._h5[dm_path]
                print(f"║      dof_maps:")
                for k in dm.keys():
                    print(f"║         • {k}: shape={dm[k].shape}, "
                          f"dtype={dm[k].dtype}")
        print("╠══ extractors: .snapshots(), .load(), .load_with_meshes()")



class EigenReader(BaseReader):
    """Modal results."""

    @property
    def kinds(self) -> list[str]:
        return list(self._h5.get("modes", {}).keys())

    def info(self) -> dict:
        return {k: dict(self._h5[f"modes/{k}"].attrs) for k in self.kinds}

    def load(self, kind: Domain | str) -> tuple[list[ModeSnapshot], Mesh, int]:
        dom  = Domain.coerce(kind)
        g    = self._h5[f"modes/{dom.value}"]

        freqs        = g["freqs"][:]
        shapes_raw   = self._decode_complex(g["shapes"][:])
        node_indexed = bool(g.attrs.get("node_indexed", False))
        attrs        = dict(g.attrs)
        mesh_id      = attrs.get("mesh_id", next(iter(self._h5["meshes"])))
        dpn          = int(attrs.get("dofs_per_node", dom.default_dpn))
        fname        = dom.field_name

        if node_indexed:
            # shapes is (n_modes, n_nodes[, dpn]) — already on mesh nodes
            # normalise to (n_modes, ...) — first dim must be n_modes
            if shapes_raw.shape[0] != len(freqs):
                shapes_raw = shapes_raw.T
            zeta  = g["zeta"][:] if "zeta" in g else [None] * len(freqs)
            mesh  = self.mesh(mesh_id, dom)
            snaps = [
                ModeSnapshot(
                    label    = f"mode {k+1} ({f_k:.1f} Hz)",
                    fields   = {fname: shapes_raw[k]},   # (n_nodes[, dpn])
                    dof_maps = {},                        # not needed
                    mesh_id  = mesh_id,
                    meta     = attrs,
                    freq     = float(f_k),
                    zeta     = zeta[k],
                    index    = k,
                )
                for k, f_k in enumerate(freqs)
            ]
        else:
            # legacy: (n_dofs, n_modes) with separate dof_map
            if shapes_raw.ndim == 2 and shapes_raw.shape[0] != len(freqs):
                shapes_raw = shapes_raw.T
            zeta    = g["zeta"][:]    if "zeta"    in g else [None] * len(freqs)
            dof_map = g["dof_map"][:] if "dof_map" in g else None
            mesh    = self.mesh(mesh_id, dom)
            snaps   = [
                ModeSnapshot(
                    label    = f"mode {k+1} ({f_k:.1f} Hz)",
                    fields   = {fname: shapes_raw[k]},
                    dof_maps = {fname: dof_map} if dof_map is not None else {},
                    mesh_id  = mesh_id,
                    meta     = attrs,
                    freq     = float(f_k),
                    zeta     = zeta[k],
                    index    = k,
                )
                for k, f_k in enumerate(freqs)
            ]

        return snaps, mesh, dpn

    def _inspect_specific(self) -> None:
        if not self.kinds:
            print("╠══ modes: (none)")
            return
        print(f"╠══ mode kinds ({len(self.kinds)}): {self.kinds}")
        for k in self.kinds:
            g       = self._h5[f"modes/{k}"]
            freqs   = g["freqs"][:]
            zeta    = g["zeta"][:] if "zeta" in g else None
            dof_map = g["dof_map"][:] if "dof_map" in g else None
            dpn     = int(g.attrs.get("dofs_per_node", Domain.coerce(k).default_dpn))
            mesh_id = g.attrs.get("mesh_id", next(iter(self._h5["meshes"])))
            shapes  = g["shapes"]

            print(f"║")
            print(f"║  ── kind = '{k}' ──────────────────────────────────────")
            print(f"║      mesh_id   : {mesh_id}")
            print(f"║      dofs/node : {dpn}")
            print(f"║      n_modes   : {len(freqs)}")
            print(f"║      shapes    : shape={shapes.shape}, dtype={shapes.dtype}")
            print(f"║                  stats: {self._stats(self._decode_complex(shapes[:]))}")
            print(f"║      freqs     : shape={freqs.shape}, dtype={freqs.dtype}")
            print(f"║                  range=[{freqs[0]:.4f}, {freqs[-1]:.4f}] Hz")
            print(f"║                  stats: {self._stats(freqs)}")
            if zeta is not None:
                print(f"║      zeta      : shape={zeta.shape}, dtype={zeta.dtype}")
                print(f"║                  stats: {self._stats(zeta)}")
            else:
                print(f"║      zeta      : (not stored)")
            if dof_map is not None:
                print(f"║      dof_map   : shape={dof_map.shape}, dtype={dof_map.dtype}")
                print(f"║                  tag range=[{int(dof_map.min())}, "
                      f"{int(dof_map.max())}]")
            else:
                print(f"║      dof_map   : (not stored)")

            # all extra datasets in this group
            extras = [n for n in g.keys()
                      if n not in {"freqs", "zeta", "shapes", "dof_map"}]
            if extras:
                print(f"║      extra datasets:")
                for n in extras:
                    d = g[n]
                    if isinstance(d, h5py.Dataset):
                        print(f"║         • {n}: shape={d.shape}, dtype={d.dtype}")

            # group attributes
            print(f"║      attrs     :")
            self._fmt_attrs(g.attrs, indent="║         ")

            # full per-mode table
            print(f"║      modes table:")
            print(f"║        {'idx':>4}  {'freq [Hz]':>14}  {'zeta':>14}")
            print(f"║        {'-'*4}  {'-'*14}  {'-'*14}")
            for i, f in enumerate(freqs):
                z_str = f"{zeta[i]:14.6g}" if zeta is not None else f"{'—':>14}"
                print(f"║        {i:>4}  {f:14.4f}  {z_str}")
        print("╠══ extractors: .kinds, .info(), .load(kind)")

class TimeReader(BaseReader):
    """Transient results."""

    def kinds(self) -> list[str]:
        return list(self._h5.get("time_histories", {}).keys())

    def runs(self, kind: str) -> list[str]:
        return sorted(self._h5[f"time_histories/{kind}"].keys())

    def _run_group(self, kind: str, run: str | int):
        runs = self.runs(kind)
        key  = runs[run] if isinstance(run, int) else run
        return self._h5[f"time_histories/{kind}/{key}"]

    def load_run(self, kind: str, run: str | int = 0) -> TimeRun:
        """Load full time history into RAM. Use iter_frames() for large runs."""
        g = self._run_group(kind, run)
        return TimeRun(
            time     = g["time"][:],
            U        = g["U"][:],
            V        = g["V"][:],
            A        = g["A"][:],
            dof_maps = {},
            mesh_id  = g.attrs["mesh_id"],
        )

    def iter_frames(self, kind: str, run: str | int = 0,
                    field: str = "u_meca", chunk_size: int = 64):
        """
        Yield (time_value, frame_array) one chunk at a time without loading
        the full U/V/A into RAM. Useful for very long runs.

            field : 'u_meca' → U,  'v_meca' → V,  'a_meca' → A
        """
        _MAP = {"u_meca": "U", "v_meca": "V", "a_meca": "A"}
        g    = self._run_group(kind, run)
        time = g["time"][:]
        ds   = g[_MAP.get(field, "U")]
        n_t  = len(time)
        for start in range(0, n_t, chunk_size):
            sl    = slice(start, min(start + chunk_size, n_t))
            block = ds[sl]                           # (chunk, n_nodes[, dpn])
            for i, frame in enumerate(block):
                yield float(time[start + i]), frame  # (n_nodes[, dpn])

    def times(self, kind: str, run: str | int = 0) -> np.ndarray:
        return self._run_group(kind, run)["time"][:]

    def mesh_id(self, kind: str, run: str | int = 0) -> str:
        return self._run_group(kind, run).attrs["mesh_id"]

    def _inspect_specific(self) -> None:
        kinds = self.kinds()
        if not kinds:
            print("╠══ time_histories: (none)")
            return
        print(f"╠══ time-history kinds ({len(kinds)}): {kinds}")
        for k in kinds:
            runs = self.runs(k)
            print(f"║")
            print(f"║  ── kind = '{k}' ({len(runs)} run(s))")
            for r in runs:
                g = self._h5[f"time_histories/{k}/{r}"]
                t = g["time"][:]
                dt = np.diff(t)
                print(f"║    ▸ run = '{r}'")
                print(f"║        mesh_id : {g.attrs.get('mesh_id', '?')}")
                print(f"║        attrs   :")
                self._fmt_attrs(g.attrs, indent="║           ")
                print(f"║        time    : n={len(t)}, "
                      f"t∈[{t[0]:.4g}, {t[-1]:.4g}] s")
                if len(dt):
                    print(f"║                  dt: min={dt.min():.4g}, "
                          f"max={dt.max():.4g}, mean={dt.mean():.4g}")
                    print(f"║                  fs≈{1.0/dt.mean():.4g} Hz")
                # arrays U / V / A and any extras
                for name in g.keys():
                    if name == "dof_maps":
                        continue
                    d = g[name]
                    if isinstance(d, h5py.Dataset):
                        arr = d[:]
                        print(f"║        {name:<7} : shape={d.shape}, "
                              f"dtype={d.dtype}")
                        print(f"║                  stats: {self._stats(arr)}")
                # dof_maps
                if "dof_maps" in g:
                    dm = g["dof_maps"]
                    print(f"║        dof_maps:")
                    for n in dm.keys():
                        dd = dm[n]
                        print(f"║           • {n}: shape={dd.shape}, "
                              f"dtype={dd.dtype}")
                        if dd.size:
                            print(f"║                tag range=["
                                  f"{int(np.min(dd[:]))}, {int(np.max(dd[:]))}]")
        print("╠══ extractors: .kinds(), .runs(kind), .load_run(kind, run), "
              ".times(), .mesh_id()")