"""Code quality (phase L): no private name imported across modules, docstrings on the public objects."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "loudpy"
DOCSTRING_COVERAGE = 1.0             # every public module, class, function, method and property (inherited ok)


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
    """(module, qualified name, has a docstring) for every module and every public class, function, method and
    property defined in the package; a method overriding a documented one inherits its docstring (inspect.getdoc,
    as Sphinx does)."""
    import importlib
    import inspect
    out = []
    for f in sorted(SRC.rglob("*.py")):
        parts = f.relative_to(SRC.parent).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        mod = importlib.import_module(name)
        out.append((name, "<module>", bool(mod.__doc__ and mod.__doc__.strip())))
        for oname, obj in vars(mod).items():
            if oname.startswith("_") or getattr(obj, "__module__", None) != name:
                continue
            if inspect.isfunction(obj):
                out.append((name, oname, bool(inspect.getdoc(obj))))
            elif inspect.isclass(obj):
                doc = (obj.__doc__ or "").strip()
                auto = doc.startswith(obj.__name__ + "(")              # the signature a dataclass gets
                out.append((name, oname, bool(doc) and not auto))
                for mname, member in vars(obj).items():
                    if mname.startswith("_"):
                        continue
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    if inspect.isfunction(member) or isinstance(member, property):
                        doc = inspect.getdoc(getattr(obj, mname, member)) if not isinstance(member, property) \
                            else (member.__doc__ or next((inspect.getdoc(getattr(b, mname)) for b in obj.__mro__[1:]
                                                          if mname in vars(b)), None))
                        out.append((name, f"{oname}.{mname}", bool(doc)))
    return out


def test_public_docstrings():
    objs = public_objects()
    documented = sum(d for *_, d in objs)
    coverage = documented / len(objs)
    missing = [f"{m}: {n}" for m, n, d in objs if not d]
    assert coverage >= DOCSTRING_COVERAGE, (f"{coverage:.1%} of {len(objs)} public objects documented; "
                                            "missing:\n" + "\n".join(missing))
