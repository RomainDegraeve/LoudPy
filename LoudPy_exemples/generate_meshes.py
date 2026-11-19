"""
generate_meshes.py — builds and saves the mesh(es) of every example without solving.

Each example script runs as written, with the solves of the studies (lp.Modes,
lp.FrequencyResponse, lp.Transient) replaced by no-ops: every model.mesh(...) happens with the
script's own sizes and paths, and FrequencyResponse still builds one mesh per band of its
remesh rule (lp.PerWavelength). The meshes are exactly the ones the examples reuse afterwards;
meshes already up to date are only checked (reused), not rebuilt.

    python3 LoudPy_exemples/generate_meshes.py            (from the repository root)
    python3 LoudPy_exemples/generate_meshes.py Eigen_study/Meca/Running_script.py   (some)
"""
import os
import sys
import time
import traceback

import numpy as np

EXAMPLES = [
    "Eigen_study/Meca/Running_script.py",
    "Eigen_study/Fourier/Running_script.py",
    "Eigen_study/FSI_DtN/Running_script.py",
    "Frequency_study/Meca/Running_scrip.py",
    "Frequency_study/FSI/Running_scrip.py",
    "Frequency_study/FSI_elec/Running_scrip.py",
    "Time_study/Multitone/Running_script.py",
]


def run(script):
    import loudpy as lp
    from loudpy.results.results import Block, ModeResults, TimeResults
    from loudpy.study.base import Study
    from loudpy.study.frequency import FrequencyResponse

    calls = []
    patches = {
        (lp.Model, "mesh"): None, (lp.Modes, "solve"): None, (lp.Transient, "solve"): None,
        (FrequencyResponse, "_solve_block"): None, (Study, "save"): None,
    }
    for (cls, name) in patches:
        patches[(cls, name)] = getattr(cls, name)
    real_mesh = patches[(lp.Model, "mesh")]

    def recording_mesh(self, path=None, **kwargs):
        kwargs.update(reuse=True, show=False)
        out = real_mesh(self, path, **kwargs)
        calls.append(str(self._mesh_path))
        return out

    lp.Model.mesh = recording_mesh
    lp.Modes.solve = lambda self, *a, **k: ModeResults([])
    lp.Transient.solve = lambda self, *a, **k: TimeResults([])
    FrequencyResponse._solve_block = lambda self, freqs, label="": (Block({}, np.asarray(freqs, float), {}), {})
    Study.save = lambda self, *a, **k: None

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, script)
    t0 = time.time()
    try:
        exec(compile(open(path).read(), path, "exec"), {"__name__": "__main__", "__file__": path})
    except Exception:
        if not calls:                       # failed before meshing: a real problem
            raise
        # after the meshes, the scripts work on results the no-op solves do not produce
        print(f"   (script stopped after meshing: {traceback.format_exc().splitlines()[-1]})")
    finally:
        for (cls, name), fn in patches.items():
            setattr(cls, name, fn)
    print(f"✓ {script}: {len(calls)} mesh(es) in {time.time() - t0:.0f} s")
    for m in calls:
        print(f"     {m}")
    return calls


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    for s in (sys.argv[1:] or EXAMPLES):
        print(f"── {s}")
        run(s)
