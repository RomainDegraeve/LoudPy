"""Generated scripts stay readable: long calls one argument per line, lists of numbers filling their lines, strings
never cut, the code unchanged."""
import ast

from loudpy.codefmt import wrap, wrap_lines


def test_long_calls_are_wrapped_and_unchanged():
    line = ("    res = study.solve(freqs=np.geomspace(20.0, 20000.0, 400), remesh=lp.PerWavelength(bands=["
            + ", ".join(f"{20.0 * 2 ** k:g}" for k in range(12)) + "], folder=HERE / 'meshes (a, b)', c=344.0))")
    out = wrap(line, 100)
    assert len(out) > 3 and all(len(x) <= 100 for x in out)
    assert out[0] == "    res = study.solve(" and out[-1] == "    )"
    assert "        freqs=np.geomspace(20.0, 20000.0, 400)," in out and "            folder=HERE / 'meshes (a, b)'," in out
    same = ast.dump(ast.parse(line.strip()))
    assert ast.dump(ast.parse("\n".join(x[4:] for x in out))) == same        # the same code
    assert wrap("x = f(1)", 100) == ["x = f(1)"]
    assert wrap_lines(["a = 1\nb = 2"]) == ["a = 1", "b = 2"]
    s = "x = '" + "y" * 120 + "'"                                            # a string is never cut
    assert wrap(s) == [s]
