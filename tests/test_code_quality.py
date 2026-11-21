"""Code quality (phase L): no private name imported across modules, docstrings on the public objects."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "loudpy"
DOCSTRING_COVERAGE = 0.75            # ratchet: raise it as docstrings are written (target 1.0, phase L)


def _modules():
    for f in sorted(SRC.rglob("*.py")):
        yield f, ast.parse(f.read_text())


def test_no_private_cross_imports():
    """`from loudpy.x import _name` from another module: the private name must be made public first."""
    found = []
    for f, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("loudpy"):
                found += [f"{f.relative_to(SRC)}: from {node.module} import {a.name}" for a in node.names
                          if a.name.startswith("_") and not a.name.startswith("__")]
    assert not found, "\n".join(found)


def public_objects():
    """(module path, qualified name, has a docstring) for every module and public class, function, method."""
    out = []
    for f, tree in _modules():
        rel = str(f.relative_to(SRC))
        out.append((rel, "<module>", ast.get_docstring(tree) is not None))

        def walk(node, prefix):
            for n in node.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not n.name.startswith("_"):
                    out.append((rel, prefix + n.name, ast.get_docstring(n) is not None))
                    if isinstance(n, ast.ClassDef):
                        walk(n, prefix + n.name + ".")
        walk(tree, "")
    return out


def test_public_docstrings():
    objs = public_objects()
    documented = sum(d for *_, d in objs)
    coverage = documented / len(objs)
    missing = [f"{m}: {n}" for m, n, d in objs if not d]
    assert coverage >= DOCSTRING_COVERAGE, (f"{coverage:.1%} of {len(objs)} public objects documented; "
                                            "missing:\n" + "\n".join(missing))
