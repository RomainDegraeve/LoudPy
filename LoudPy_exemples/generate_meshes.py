"""
generate_meshes.py — builds and saves the mesh(es) of every example without solving.

Each example script is run as written, with the study classes (EigenStudy, FreqStudy,
TimeStudy) replaced by do-nothing stand-ins: every problem.mesh(..., reuse_mesh=True) call
happens with the script's own sizes, PML settings and msh_path (one per remesh block for the
frequency sweeps), so the meshes are exactly the ones the examples reuse afterwards.
Meshes already up to date are only checked (reused), not rebuilt.

    python3 LoudPy_exemples/generate_meshes.py            (from the repository root)
    python3 LoudPy_exemples/generate_meshes.py Eigen_study/Meca/Running_script.py   (some)
"""
import os
import sys
import time
import traceback

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

EXAMPLES = [
    "Eigen_study/Meca/Running_script.py",
    "Eigen_study/FSI_DtN/Running_script.py",
    "Frequency_study/Meca/Running_scrip.py",
    "Frequency_study/FSI/Running_scrip.py",
    "Frequency_study/FSI_elec/Running_scrip.py",
    "Time_study/Multitone/Running_script.py",
]


class _NoStudy:
    """Stand-in for a study: accepts every call, solves and saves nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _MeshingDone(Exception):
    pass


def run(script):
    import loudpy.Studies as studies
    from loudpy.Studies import Problem
    saved = {n: getattr(studies, n) for n in ("EigenStudy", "FreqStudy", "TimeStudy")
             if hasattr(studies, n)}
    mesh = Problem.mesh
    calls = []

    def recording_mesh(self, *args, **kwargs):
        kwargs["show_mesh_gui"] = False
        kwargs["reuse_mesh"] = True
        mesh(self, *args, **kwargs)
        calls.append(self.msh_path)

    for n in saved:
        setattr(studies, n, _NoStudy)
    Problem.mesh = recording_mesh
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, script)
    t0 = time.time()
    try:
        exec(compile(open(path).read(), path, "exec"), {"__name__": "__main__", "__file__": path})
    except Exception:
        if not calls:                       # failed before meshing: a real problem
            raise
        # after the meshes, the scripts work on results the stand-ins do not produce
        print(f"   (script stopped after meshing: {traceback.format_exc().splitlines()[-1]})")
    finally:
        for n, cls in saved.items():
            setattr(studies, n, cls)
        Problem.mesh = mesh
    print(f"✓ {script}: {len(calls)} mesh(es) in {time.time() - t0:.0f} s")
    for m in calls:
        print(f"     {m}")
    return calls


if __name__ == "__main__":
    for s in (sys.argv[1:] or EXAMPLES):
        print(f"── {s}")
        run(s)
