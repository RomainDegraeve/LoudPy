"""
The electrical side of a loudspeaker as a circuit: an amplifier, ideal filter blocks (EQ, crossovers), passive parts
and the drivers' coils, on named nodes ("0" is the ground). It drives a scene (lp.FrequencyResponse(scene,
drive=circuit)) or a 2D model: solved by modified nodal analysis (MNA) inside the coupled system, its node voltages
and branch currents a few more unknowns eliminated with the modes at each frequency — each coil's back-EMF Bl·v_coil
in its branch, its current's force Bl·I on its driver.

    import loudpy as lp
    from loudpy.circuit import Source, Filter, Coil, Resistor, LR2, linkwitz_riley, peq
    c = lp.Circuit(Source("amp", "in", U=2.83),
                   Filter("lp", "in", "w", linkwitz_riley(4, 2000.0, "low") * peq(80.0, 3.0, 1.0)),
                   Resistor("Rs", "w", "c"), Coil("main", "c", "0", driver="main", model=LR2(Re=5.6, Le=3e-4,
                   R2=2.0, L2=1e-4, Bl=6.1)))
    res = lp.FrequencyResponse(scene, drive=c).solve(freqs)
    res.voltage("c"), res.current("main"), res.input_impedance(), res.impedance("main")

Unknowns x = [node voltages (ground left out), branch currents]; a branch current flows from the part's first node
to its second through the part (a source's: the current it delivers out of its first node). Rows:
    R, C          admittance stamps 1/R, sC
    L             v_a − v_b − sL i = 0
    Source        v_a − v_b + R_out i = U           (an amplifier: U behind its output resistance)
    Filter        v_out − H(s) v_in = 0             (an ideal block, both voltages to ground)
    Coil          Z(s) i + Bl v_coil − (v_a − v_b) = 0, v_coil = s fᵀ q (its driver's modes)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from loudpy.errors import ModelError

GROUND = "0"


# ── transfer functions of the ideal filter blocks ────────────────────────────────────────────────────────────────
@dataclass
class Transfer:
    """
    H(s) = gain · Π num_i(s) / den_i(s) · e^{−s·delay}: polynomial sections in s (coefficients from the highest
    power down). Cascaded by multiplication: linkwitz_riley(4, 2e3) * peq(80, 3, 1).
    """
    num: list = field(default_factory=lambda: [[1.0]])
    den: list = field(default_factory=lambda: [[1.0]])
    gain: float = 1.0
    delay: float = 0.0

    def __call__(self, s):
        h = complex(self.gain) * np.exp(-s * self.delay)
        for n, d in zip(self.num, self.den):
            h = h * np.polyval(n, s) / np.polyval(d, s)
        return h

    def __mul__(self, other: "Transfer") -> "Transfer":
        return Transfer(self.num + other.num, self.den + other.den, self.gain * other.gain, self.delay + other.delay)

    def response(self, freqs) -> np.ndarray:
        """H(jω) at the frequencies [Hz]."""
        return np.array([self(2j * np.pi * f) for f in np.atleast_1d(freqs)])


def _w(f):
    return 2 * math.pi * float(f)


def gain(db: float) -> Transfer:
    """A gain [dB]."""
    return Transfer(gain=10 ** (db / 20))


def delay(t: float) -> Transfer:
    """A pure delay [s]."""
    return Transfer(delay=float(t))


def biquad(num, den) -> Transfer:
    """A section num(s)/den(s), coefficients from the highest power of s down."""
    return Transfer([list(map(float, num))], [list(map(float, den))])


def butterworth(order: int, fc: float, kind: str = "low") -> Transfer:
    """A Butterworth low or high pass ("low" / "high") of `order` at fc [Hz] (−3 dB there)."""
    if kind not in ("low", "high"):
        raise ModelError(f"butterworth: kind 'low' or 'high', got {kind!r}")
    if int(order) < 1:
        raise ModelError("butterworth: order ≥ 1")
    w = _w(fc)
    num, den = [], []
    n = int(order)
    for k in range(n // 2):                                       # conjugate pole pairs: s² + 2 sin(θ) w s + w²
        theta = math.pi * (2 * k + 1) / (2 * n)
        den.append([1.0, 2 * math.sin(theta) * w, w * w])
        num.append([w * w] if kind == "low" else [1.0, 0.0, 0.0])
    if n % 2:                                                     # the real pole
        den.append([1.0, w])
        num.append([w] if kind == "low" else [1.0, 0.0])
    return Transfer(num, den)


def linkwitz_riley(order: int, fc: float, kind: str = "low") -> Transfer:
    """A Linkwitz–Riley low or high pass of even `order` (2, 4, 8) at fc [Hz]: two Butterworths of half the order
    (−6 dB at fc; the low and high pass sum flat, in phase for order 4)."""
    if int(order) % 2 or int(order) < 2:
        raise ModelError("linkwitz_riley: an even order (2, 4, 8 …)")
    b = butterworth(int(order) // 2, fc, kind)
    return b * b


def peq(f0: float, gain_db: float, q: float) -> Transfer:
    """A parametric EQ (peaking) at f0 [Hz], gain [dB], quality factor q (the analog RBJ form)."""
    a, w = 10 ** (gain_db / 40), _w(f0)
    return biquad([1.0, a * w / q, w * w], [1.0, w / (a * q), w * w])


def low_shelf(f0: float, gain_db: float, q: float = 1 / math.sqrt(2)) -> Transfer:
    """A low shelf at f0 [Hz], gain [dB] below it (the analog RBJ form)."""
    a, w = 10 ** (gain_db / 40), _w(f0)
    r = math.sqrt(a) / q * w
    return Transfer([[a, a * r, a * a * w * w]], [[a, r, w * w]])


def high_shelf(f0: float, gain_db: float, q: float = 1 / math.sqrt(2)) -> Transfer:
    """A high shelf at f0 [Hz], gain [dB] above it (the analog RBJ form)."""
    a, w = 10 ** (gain_db / 40), _w(f0)
    r = math.sqrt(a) / q * w
    return Transfer([[a * a, a * r, a * w * w]], [[1.0, r, a * w * w]])


# ── coil models (the magnetic side): Z(s) of the blocked coil, and its force factor ─────────────────────────────────
@dataclass
class Semi:
    """Re + (s)ⁿ Le: a resistance and a (semi-)inductance (n = 1: an inductance; n < 1: eddy-current losses)."""
    Re: float
    Bl: float
    Le: float = 0.0
    n: float = 1.0

    def impedance(self, s):
        """The blocked coil's impedance at s = jω [Ω]."""
        return self.Re + s ** self.n * self.Le


@dataclass
class Leach:
    """Leach's model: Re + K (s)ⁿ (the semi-inductance under its usual names)."""
    Re: float
    Bl: float
    K: float = 0.0
    n: float = 0.7

    def impedance(self, s):
        """The blocked coil's impedance at s = jω [Ω]."""
        return self.Re + self.K * s ** self.n


@dataclass
class LR2:
    """The LR-2 model: Re + s Le + (R2 ∥ s L2) — the coil inductance and the eddy-current branch."""
    Re: float
    Bl: float
    Le: float = 0.0
    R2: float = 0.0
    L2: float = 0.0

    def impedance(self, s):
        """The blocked coil's impedance at s = jω [Ω]."""
        par = 0.0 if self.R2 == 0 or self.L2 == 0 else self.R2 * s * self.L2 / (self.R2 + s * self.L2)
        return self.Re + s * self.Le + par


@dataclass
class Wright:
    """Wright's empirical model: Re + Krm ω^Erm + j Kxm ω^Exm (ω = 2πf)."""
    Re: float
    Bl: float
    Krm: float = 0.0
    Erm: float = 1.0
    Kxm: float = 0.0
    Exm: float = 1.0

    def impedance(self, s):
        """The blocked coil's impedance at s = jω [Ω]."""
        w = abs(complex(s).imag)
        return self.Re + self.Krm * w ** self.Erm + 1j * self.Kxm * w ** self.Exm


COIL_MODELS = (Semi, Leach, LR2, Wright)


# ── the parts ────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Resistor:
    """A resistance R [Ω] between nodes a and b."""
    name: str
    a: str
    b: str
    R: float


@dataclass
class Inductor:
    """An inductance L [H] between nodes a and b (its current a → b an unknown)."""
    name: str
    a: str
    b: str
    L: float


@dataclass
class Capacitor:
    """A capacitance C [F] between nodes a and b."""
    name: str
    a: str
    b: str
    C: float


@dataclass
class Source:
    """The amplifier: a voltage U [V] (complex amplitude) between a (+) and b (−) behind its output resistance R_out
    [Ω]; its current is the one it delivers out of a."""
    name: str
    a: str
    b: str = GROUND
    U: complex = 1.0
    R_out: float = 0.0


@dataclass
class Filter:
    """An ideal filter block (an active or DSP filter): v(out) = H(s) v(inp), both to ground; it drives `out`
    whatever the load."""
    name: str
    inp: str
    out: str
    H: Transfer = field(default_factory=Transfer)


@dataclass
class Coil:
    """A driver's coil between nodes a and b: its blocked impedance and force factor (model: Semi, Leach, LR2,
    Wright) and its driver (a scene's driver by name; None: the coil of a 2D model)."""
    name: str
    a: str
    b: str
    model: Any
    driver: str | None = None


PARTS = (Resistor, Inductor, Capacitor, Source, Filter, Coil)
_BRANCH = (Inductor, Source, Filter, Coil)                              # the parts whose current is an unknown


def _nodes(p) -> tuple:
    return (p.inp, p.out) if isinstance(p, Filter) else (p.a, p.b)


class Circuit:
    """The electrical side of a loudspeaker (module docstring): parts on named nodes, "0" the ground."""

    def __init__(self, *parts):
        self.parts: list = []
        self.add(*parts)

    def add(self, *parts) -> "Circuit":
        """Add parts (or lists of parts); each name once, a coil with a known model."""
        for p in parts:
            if isinstance(p, (list, tuple)):
                self.add(*p)
                continue
            if not isinstance(p, PARTS):
                raise ModelError(f"a circuit part is one of {[t.__name__ for t in PARTS]}, got {p!r}")
            if any(q.name == p.name for q in self.parts):
                raise ModelError(f"a part is already named {p.name!r}")
            if isinstance(p, Coil) and not isinstance(p.model, COIL_MODELS):
                raise ModelError(f"coil {p.name}: its model is one of {[t.__name__ for t in COIL_MODELS]}")
            self.parts.append(p)
        return self

    def __getitem__(self, name):
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(f"no part {name!r} (parts: {[p.name for p in self.parts]})")

    def __repr__(self):
        return f"Circuit({', '.join(f'{type(p).__name__} {p.name}' for p in self.parts)})"

    # ── its unknowns ────────────────────────────────────────────────────────────────────────────────────────────
    @property
    def nodes(self) -> list[str]:
        """The nodes with an unknown voltage (every node but the ground), in the order they appear."""
        out: list[str] = []
        for p in self.parts:
            for n in _nodes(p):
                if n != GROUND and n not in out:
                    out.append(str(n))
        return out

    @property
    def branches(self) -> list[str]:
        """The parts whose current is an unknown (inductors, sources, filters, coils)."""
        return [p.name for p in self.parts if isinstance(p, _BRANCH)]

    @property
    def coils(self) -> list:
        """The coil parts (one per driver it drives)."""
        return [p for p in self.parts if isinstance(p, Coil)]

    @property
    def size(self) -> int:
        """The number of unknowns: node voltages and branch currents."""
        return len(self.nodes) + len(self.branches)

    def index(self, name: str) -> int:
        """The unknown of a node (its voltage) or of a branch part (its current)."""
        nodes, branches = self.nodes, self.branches
        if name in nodes:
            return nodes.index(name)
        if name in branches:
            return len(nodes) + branches.index(name)
        raise ModelError(f"{name!r} is neither a node nor a part with its own current (nodes {nodes}, branch parts "
                         f"{branches})")

    def check(self) -> None:
        """Refuses a circuit that cannot be solved: no source, a node joined to nothing else, no path to ground."""
        if not any(isinstance(p, Source) for p in self.parts):
            raise ModelError("a circuit needs a Source (the amplifier)")
        uses: dict = {}
        for p in self.parts:
            for n in _nodes(p):
                uses[n] = uses.get(n, 0) + 1
        loose = [n for n, k in uses.items() if k < 2 and n != GROUND]
        if loose:
            raise ModelError(f"node(s) {loose} joined to one part only: wire them, or ground them ('0')")
        if GROUND not in uses:
            raise ModelError("no part is connected to the ground '0'")

    # ── modified nodal analysis ─────────────────────────────────────────────────────────────────────────────────
    def matrix(self, s) -> np.ndarray:
        """The MNA matrix at s (its coils blocked: their back-EMF is the coupling to the mechanics)."""
        n = self.size
        A = np.zeros((n, n), complex)
        node = {nm: i for i, nm in enumerate(self.nodes)}

        def stamp(a, b, y):
            for i, j, sign in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
                if i in node and j in node:
                    A[node[i], node[j]] += sign * y

        def incidence(a, b, k, sign=1.0):                          # the branch current leaves a, enters b
            if a in node:
                A[node[a], k] += sign
            if b in node:
                A[node[b], k] -= sign

        def drop(a, b, k):                                         # v_a − v_b in the branch row
            if a in node:
                A[k, node[a]] += 1.0
            if b in node:
                A[k, node[b]] -= 1.0
        for p in self.parts:
            if isinstance(p, Resistor):
                stamp(p.a, p.b, 1.0 / p.R)
            elif isinstance(p, Capacitor):
                stamp(p.a, p.b, s * p.C)
            elif isinstance(p, Inductor):
                k = self.index(p.name)
                incidence(p.a, p.b, k)
                drop(p.a, p.b, k)
                A[k, k] -= s * p.L
            elif isinstance(p, Source):
                k = self.index(p.name)
                incidence(p.a, p.b, k, -1.0)                       # delivered out of a
                drop(p.a, p.b, k)
                A[k, k] += p.R_out
            elif isinstance(p, Filter):
                k = self.index(p.name)
                if p.out in node:
                    A[node[p.out], k] -= 1.0                       # driven into its output node
                    A[k, node[p.out]] += 1.0
                if p.inp in node:
                    A[k, node[p.inp]] -= p.H(s)
            elif isinstance(p, Coil):
                k = self.index(p.name)
                incidence(p.a, p.b, k)
                drop(p.a, p.b, k)
                A[k, :] *= -1.0                                    # Z i − (v_a − v_b) (+ Bl v_coil: the coupling)
                A[k, k] += p.model.impedance(s)
        return A

    def rhs(self, s=None) -> np.ndarray:
        """The sources' voltages in their rows."""
        b = np.zeros(self.size, complex)
        for p in self.parts:
            if isinstance(p, Source):
                b[self.index(p.name)] = complex(p.U)
        return b

    def coupling(self, forces: dict) -> tuple[np.ndarray, np.ndarray]:
        """(M_qe, M_eq) coupling the circuit to modal coordinates: forces {coil name: its unit coil force on the
        modes (n_y,)}; q row −Bl f i, coil row + Bl fᵀ (times s: the coil velocity)."""
        n_y = len(next(iter(forces.values()))) if forces else 0
        qe = np.zeros((n_y, self.size))
        eq = np.zeros((self.size, n_y))
        for c in self.coils:
            if c.name not in forces:
                continue
            k = self.index(c.name)
            f = np.asarray(forces[c.name], float)
            qe[:, k] = -c.model.Bl * f
            eq[k, :] = c.model.Bl * f
        return qe, eq

    # ── as data (results, documents) ─────────────────────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """The circuit as plain data (JSON): its parts with their values."""
        def value(v):
            if isinstance(v, Transfer):
                return {"num": v.num, "den": v.den, "gain": v.gain, "delay": v.delay}
            if isinstance(v, COIL_MODELS):
                return {"type": type(v).__name__, **{k: getattr(v, k) for k in v.__dataclass_fields__}}
            if isinstance(v, complex):
                return [v.real, v.imag]
            return v
        return {"parts": [{"type": type(p).__name__, **{k: value(getattr(p, k)) for k in p.__dataclass_fields__}}
                          for p in self.parts]}

    @classmethod
    def from_dict(cls, data: dict) -> "Circuit":
        """The circuit of to_dict."""
        types = {t.__name__: t for t in PARTS}
        models = {t.__name__: t for t in COIL_MODELS}
        parts = []
        for d in data.get("parts", []):
            d = dict(d)
            t = types[d.pop("type")]
            if "H" in d and isinstance(d["H"], dict):
                d["H"] = Transfer(**d["H"])
            if "model" in d and isinstance(d["model"], dict):
                m = dict(d["model"])
                d["model"] = models[m.pop("type")](**m)
            if "U" in d and isinstance(d["U"], (list, tuple)):
                d["U"] = complex(*d["U"])
            parts.append(t(**d))
        return cls(*parts)

    # ── alone (no mechanics: blocked coils) ─────────────────────────────────────────────────────────────────────
    def solve(self, freqs) -> "CircuitResponse":
        """The circuit alone at the frequencies [Hz], its coils blocked (no back-EMF): the filters, the loads."""
        self.check()
        freqs = np.atleast_1d(np.asarray(freqs, float))
        x = np.array([np.linalg.solve(self.matrix(2j * np.pi * f), self.rhs()) for f in freqs])
        return CircuitResponse(self, freqs, x)


class CircuitResponse:
    """The node voltages and branch currents of a circuit over frequency (alone, or from a coupled study)."""

    def __init__(self, circuit: Circuit, freqs, x):
        self.circuit, self.frequencies, self.x = circuit, np.asarray(freqs, float), np.asarray(x, complex)

    def _data(self, values, quantity, **meta):
        from loudpy.results.data import Data
        return Data(np.asarray(values), {"f": self.frequencies}, quantity, meta=meta)

    def _v(self, node) -> np.ndarray:
        return np.zeros(len(self.frequencies), complex) if node == GROUND else self.x[:, self.circuit.index(node)]

    def voltage(self, node: str, ref: str = GROUND):
        """The voltage of a node (to `ref`, default the ground) [V]."""
        return self._data(self._v(node) - self._v(ref), "voltage", at=f"{node} − {ref}")

    def current(self, part: str):
        """The current through a part [A] (a source: the current it delivers)."""
        p = self.circuit[part]
        s = 2j * np.pi * self.frequencies
        if isinstance(p, _BRANCH):
            i = self.x[:, self.circuit.index(p.name)]
        elif isinstance(p, Resistor):
            i = (self._v(p.a) - self._v(p.b)) / p.R
        else:
            i = s * p.C * (self._v(p.a) - self._v(p.b))
        return self._data(i, "current", at=part)

    def impedance(self, part: str):
        """The impedance across a part: its voltage drop / its current [Ω] (a coil: its whole input impedance,
        motional part included in a coupled study)."""
        p = self.circuit[part]
        a, b = _nodes(p) if not isinstance(p, Filter) else (p.out, GROUND)
        return self._data((self._v(a) - self._v(b)) / self.current(part).values, "Z", at=part)

    def input_impedance(self, source: str | None = None):
        """The impedance the amplifier sees at its terminals [Ω]."""
        src = [p for p in self.circuit.parts if isinstance(p, Source)]
        p = src[0] if source is None else self.circuit[source]
        return self._data((self._v(p.a) - self._v(p.b)) / self.x[:, self.circuit.index(p.name)], "Z", at=p.name)

    def transfer(self, node: str, source: str | None = None):
        """A node's voltage over its source's U (a filter's or a network's transfer function)."""
        src = [p for p in self.circuit.parts if isinstance(p, Source)]
        p = src[0] if source is None else self.circuit[source]
        return self._data(self._v(node) / complex(p.U), "transfer", at=node)
