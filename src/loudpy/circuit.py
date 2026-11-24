"""
The electrical side of a loudspeaker as a circuit: an amplifier, ideal filter blocks (EQ, crossovers), passive parts
and the drivers' coils, on named nodes ("0" is the ground). It drives a scene (lp.FrequencyResponse(scene,
drive=circuit)) or a 2D model: solved by modified nodal analysis (MNA) on top of the structure and the air, which are
solved once per coil for a unit force on it — each coil's back-EMF Bl·v_coil from the coils' mobility in its branch,
its current's force Bl·I weighting its driver's unit response (loudpy.physics.electric). The circuit of a result can
change afterwards: res.redrive(other_circuit).

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
    parts: list = field(default_factory=list)              # the stages it was made of: [{"type", "params"}]

    def __call__(self, s):
        h = complex(self.gain) * np.exp(-s * self.delay)
        for n, d in zip(self.num, self.den):
            h = h * np.polyval(n, s) / np.polyval(d, s)
        return h

    def __mul__(self, other: "Transfer") -> "Transfer":
        return Transfer(self.num + other.num, self.den + other.den, self.gain * other.gain, self.delay + other.delay,
                        self.parts + other.parts)

    def response(self, freqs) -> np.ndarray:
        """H(jω) at the frequencies [Hz]."""
        return np.array([self(2j * np.pi * f) for f in np.atleast_1d(freqs)])


def _w(f):
    return 2 * math.pi * float(f)


def _stage(tf: Transfer, name: str, **params) -> Transfer:
    """A transfer function that remembers the stage it is (its block: loudpy.schema)."""
    tf.parts = [{"type": f"circuit.{name}", "params": params}]
    return tf


def gain(db: float) -> Transfer:
    """A gain [dB]."""
    return _stage(Transfer(gain=10 ** (db / 20)), "gain", db=db)


def delay(t: float) -> Transfer:
    """A pure delay [s]."""
    return _stage(Transfer(delay=float(t)), "delay", t=t)


def biquad(num, den) -> Transfer:
    """A section num(s)/den(s), coefficients from the highest power of s down."""
    num, den = list(map(float, num)), list(map(float, den))
    return _stage(Transfer([num], [den]), "biquad", num=num, den=den)


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
    return _stage(Transfer(num, den), "butterworth", order=n, fc=fc, kind=kind)


def linkwitz_riley(order: int, fc: float, kind: str = "low") -> Transfer:
    """A Linkwitz–Riley low or high pass of even `order` (2, 4, 8) at fc [Hz]: two Butterworths of half the order
    (−6 dB at fc; the low and high pass sum flat, in phase for order 4)."""
    if int(order) % 2 or int(order) < 2:
        raise ModelError("linkwitz_riley: an even order (2, 4, 8 …)")
    b = butterworth(int(order) // 2, fc, kind)
    return _stage(b * b, "linkwitz_riley", order=int(order), fc=fc, kind=kind)


def peq(f0: float, gain_db: float, q: float) -> Transfer:
    """A parametric EQ (peaking) at f0 [Hz], gain [dB], quality factor q (the analog RBJ form)."""
    a, w = 10 ** (gain_db / 40), _w(f0)
    return _stage(Transfer([[1.0, a * w / q, w * w]], [[1.0, w / (a * q), w * w]]), "peq", f0=f0, gain_db=gain_db,
                  q=q)


def low_shelf(f0: float, gain_db: float, q: float = 1 / math.sqrt(2)) -> Transfer:
    """A low shelf at f0 [Hz], gain [dB] below it (the analog RBJ form)."""
    a, w = 10 ** (gain_db / 40), _w(f0)
    r = math.sqrt(a) / q * w
    return _stage(Transfer([[a, a * r, a * a * w * w]], [[a, r, w * w]]), "low_shelf", f0=f0, gain_db=gain_db, q=q)


def high_shelf(f0: float, gain_db: float, q: float = 1 / math.sqrt(2)) -> Transfer:
    """A high shelf at f0 [Hz], gain [dB] above it (the analog RBJ form)."""
    a, w = 10 ** (gain_db / 40), _w(f0)
    r = math.sqrt(a) / q * w
    return _stage(Transfer([[a * a, a * r, a * w * w]], [[1.0, r, a * w * w]]), "high_shelf", f0=f0,
                  gain_db=gain_db, q=q)


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


def motor_to_dict(m) -> dict | None:
    """A coil model (a driver's motor) as plain data: {"type": "Semi", "Re": …, …}; None stays None."""
    if m is None:
        return None
    return {"type": type(m).__name__, **{k: getattr(m, k) for k in m.__dataclass_fields__}}


def motor_from_dict(d):
    """The coil model of motor_to_dict (also a result file's description of it); None stays None."""
    if d is None or isinstance(d, COIL_MODELS):
        return d
    models = {t.__name__: t for t in COIL_MODELS}
    d = dict(d)
    kind = str(d.pop("type")).split(".")[-1]
    if kind not in models:
        raise ModelError(f"a motor is one of {list(models)}, got {kind!r}")
    return models[kind](**{k: v for k, v in d.items() if k in models[kind].__dataclass_fields__})


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

    def __post_init__(self):
        self.U = complex(self.U)


@dataclass
class Filter:
    """An ideal filter block (an active or DSP filter): v(out) = H(s) v(inp), both to ground; it drives `out`
    whatever the load."""
    name: str
    inp: str
    out: str
    H: Any = field(default_factory=Transfer)

    def __post_init__(self):
        if isinstance(self.H, (list, tuple)):                      # its stages (blocks): their product
            h = self.H[0] if self.H else Transfer()
            for t in self.H[1:]:
                h = h * t
            self.H = h
        if not isinstance(self.H, Transfer):
            raise ModelError(f"filter {self.name}: H is a transfer function (loudpy.circuit: butterworth, "
                             f"linkwitz_riley, peq …), got {self.H!r}")


@dataclass
class Coil:
    """A driver's coil between nodes a and b, and its driver (a scene's driver by name; None: the coil of a 2D model).
    Its motor — the blocked impedance and force factor (Semi, Leach, LR2, Wright) — is the driver's (its lp.Coil
    item's motor=, or Driver.from_modes(motor=)) unless `model` gives one here."""
    name: str
    a: str
    b: str
    model: Any = None
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
            if isinstance(p, Coil) and p.model is not None and not isinstance(p.model, COIL_MODELS):
                raise ModelError(f"coil {p.name}: its model is one of {[t.__name__ for t in COIL_MODELS]} (or none: "
                                 "its driver's motor)")
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

    def check(self, freqs=None) -> None:
        """
        Refuses a circuit that cannot be solved, saying why: no source; a node joined to one part only; two parts
        setting the voltage of one node (sources and filter outputs are ideal: each fixes its node's voltage, two on a
        node contradict each other); a node with no path to the ground through parts carrying a current (a filter's
        input draws none); equations with no single solution (checked at `freqs` [Hz], default 20 Hz – 20 kHz).
        """
        if not any(isinstance(p, Source) for p in self.parts):
            raise ModelError("a circuit needs a Source (the amplifier)")
        bare = [c.name for c in self.coils if c.model is None]
        if bare:
            raise ModelError(f"coil(s) {bare}: no motor — give the driver one (its lp.Coil item: motor=…, or "
                             "Driver.from_modes(motor=…)) or the coil a model=")
        uses: dict = {}
        for p in self.parts:
            for n in _nodes(p):
                uses[n] = uses.get(n, 0) + 1
        loose = [n for n, k in uses.items() if k < 2 and n != GROUND]
        if loose:
            raise ModelError(f"node(s) {loose} joined to one part only: wire them, or ground them ('0')")
        if GROUND not in uses:
            raise ModelError("no part is connected to the ground '0'")
        self._check_setters()
        self._check_grounded()
        freqs = np.geomspace(20.0, 20000.0, 4) if freqs is None else np.atleast_1d(np.asarray(freqs, float))
        pick = freqs[np.unique(np.linspace(0, len(freqs) - 1, min(len(freqs), 5)).round().astype(int))]
        for f in pick:
            A = self.matrix(2j * np.pi * f)
            scale = np.abs(A).max(axis=1, keepdims=True)
            scale[scale == 0] = 1.0
            with np.errstate(all="ignore"):
                cond = np.linalg.cond(A / scale)
            if not np.isfinite(cond) or cond > 1e14:
                raise ModelError(f"the circuit's equations have no single solution at {f:.4g} Hz (a loop of ideal "
                                 "parts, or a part of the circuit left floating): check its wiring")

    def _check_setters(self) -> None:
        """Sources and filter outputs fix a voltage: two on one node (or a loop of them) contradict each other."""
        parent: dict = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        done: list = []
        for p in self.parts:
            if isinstance(p, Source):
                a, b = p.a, p.b
            elif isinstance(p, Filter):
                a, b = p.out, GROUND
            else:
                continue
            if find(a) == find(b):
                root = find(a)
                others = [q.name for q, x in done if find(x) == root]
                node = a if a != GROUND else b
                raise ModelError(
                    f"{' and '.join(others[-1:] + [p.name])} both set the voltage of node {node!r}: sources and filter "
                    "outputs are ideal, each fixes its node's voltage, so two on one node contradict each other — "
                    "chain the stages in one Filter (H=[…, …]), or join two outputs through resistors")
            parent[find(a)] = find(b)
            done.append((p, a))

    def _check_grounded(self) -> None:
        """Every node reaches the ground through parts that carry a current (a filter: its output to ground)."""
        links: dict = {}
        for p in self.parts:
            a, b = (p.out, GROUND) if isinstance(p, Filter) else (p.a, p.b)
            links.setdefault(a, set()).add(b)
            links.setdefault(b, set()).add(a)
        seen, todo = {GROUND}, [GROUND]
        while todo:
            for m in links.get(todo.pop(), ()):
                if m not in seen:
                    seen.add(m)
                    todo.append(m)
        floating = [n for n in self.nodes if n not in seen]
        if floating:
            raise ModelError(f"node(s) {floating} have no path to the ground '0' through parts carrying a current (a "
                             "filter's input draws none): connect that part of the circuit to the ground")

    def with_motors(self, motors: dict | None) -> "Circuit":
        """This circuit with each coil without a model given its driver's motor (motors: {driver name, or None for a
        2D model's coil: a coil model}); the coils given a model keep it."""
        import dataclasses
        if not motors:
            return self
        parts = [dataclasses.replace(p, model=motors.get(p.driver)) if isinstance(p, Coil) and p.model is None
                 else p for p in self.parts]
        return Circuit(*parts)

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
                return {"num": v.num, "den": v.den, "gain": v.gain, "delay": v.delay, "parts": v.parts}
            if isinstance(v, COIL_MODELS):
                return motor_to_dict(v)
            if isinstance(v, complex):
                return [v.real, v.imag]
            return v
        return {"parts": [{"type": type(p).__name__, **{k: value(getattr(p, k)) for k in p.__dataclass_fields__}}
                          for p in self.parts]}

    @classmethod
    def from_dict(cls, data: dict) -> "Circuit":
        """The circuit of to_dict."""
        types = {t.__name__: t for t in PARTS}
        parts = []
        for d in data.get("parts", []):
            d = dict(d)
            t = types[d.pop("type")]
            if "H" in d and isinstance(d["H"], dict):
                d["H"] = Transfer(**d["H"])
            if "model" in d and isinstance(d["model"], dict):
                d["model"] = motor_from_dict(d["model"])
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
        """The impedance the amplifier sees at its terminals [Ω] (infinite where it delivers no current, e.g. into an
        ideal filter block's input)."""
        src = [p for p in self.circuit.parts if isinstance(p, Source)]
        p = src[0] if source is None else self.circuit[source]
        i = self.x[:, self.circuit.index(p.name)]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(i != 0, (self._v(p.a) - self._v(p.b)) / np.where(i != 0, i, 1.0), np.inf)
        return self._data(z, "Z", at=p.name)

    def transfer(self, node: str, source: str | None = None):
        """A node's voltage over its source's U (a filter's or a network's transfer function)."""
        src = [p for p in self.circuit.parts if isinstance(p, Source)]
        p = src[0] if source is None else self.circuit[source]
        return self._data(self._v(node) / complex(p.U), "transfer", at=node)


# ── SPICE netlists ───────────────────────────────────────────────────────────────────────────────────────────────
_SI = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "mil": 25.4e-6, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9,
       "p": 1e-12, "f": 1e-15}


def spice_value(text: str) -> float:
    """A SPICE number with its scale suffix and unit ("4.7k", "10uF", "1.5mH", "2Meg") → a float."""
    import re
    m = re.match(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(meg|mil|[tgkmunpfµ])?", str(text).strip(), re.I)
    if m is None:
        raise ModelError(f"{text!r} is not a SPICE number")
    return float(m.group(1)) * _SI.get((m.group(2) or "").lower(), 1.0)


def _laplace(expr: str) -> Transfer:
    """A rational function of s written as an expression (LTspice's Laplace=…: + − * / ** ^, numbers, s)."""
    import ast
    P = np.polynomial.polynomial

    def rat(node):
        if isinstance(node, ast.Expression):
            return rat(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return np.array([float(node.value)]), np.array([1.0])
        if isinstance(node, ast.Name) and node.id.lower() == "s":
            return np.array([0.0, 1.0]), np.array([1.0])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            n, d = rat(node.operand)
            return (-n if isinstance(node.op, ast.USub) else n), d
        if isinstance(node, ast.BinOp):
            (an, ad), op = rat(node.left), node.op
            if isinstance(op, ast.Pow):
                k = node.right
                if not (isinstance(k, ast.Constant) and float(k.value).is_integer() and k.value >= 0):
                    raise ModelError(f"Laplace={expr}: a power must be a whole number ≥ 0")
                return P.polypow(an, int(k.value)), P.polypow(ad, int(k.value))
            bn, bd = rat(node.right)
            if isinstance(op, ast.Add):
                return P.polyadd(P.polymul(an, bd), P.polymul(bn, ad)), P.polymul(ad, bd)
            if isinstance(op, ast.Sub):
                return P.polysub(P.polymul(an, bd), P.polymul(bn, ad)), P.polymul(ad, bd)
            if isinstance(op, ast.Mult):
                return P.polymul(an, bn), P.polymul(ad, bd)
            if isinstance(op, ast.Div):
                return P.polymul(an, bd), P.polymul(ad, bn)
        raise ModelError(f"Laplace={expr}: only numbers, s, + − * / and whole powers")
    num, den = rat(ast.parse(expr.replace("^", "**"), mode="eval"))
    return biquad(list(num[::-1]), list(den[::-1]))


def _poly_text(c) -> str:
    """A polynomial in s (coefficients from the highest power) as an expression."""
    n = len(c) - 1
    terms = [f"{float(a):.12g}" + ("" if k == n else "*s" if n - k == 1 else f"*s**{n - k}")
             for k, a in enumerate(c) if a != 0]
    return "(" + ("+".join(terms).replace("+-", "-") or "0") + ")"


def _laplace_text(H: Transfer) -> str:
    out = [f"{H.gain:.12g}"] + [f"{_poly_text(n)}/{_poly_text(d)}" for n, d in zip(H.num, H.den)]
    if H.delay:
        out.append(f"exp(-s*{H.delay:.12g})")
    return "*".join(out)


def to_spice(circuit: "Circuit", title: str = "loudpy circuit") -> str:
    """
    The circuit as a SPICE netlist (LTspice reads it): R, L, C as they are; the amplifier as V (AC) behind its
    output resistance; a filter block as a Laplace voltage source (E … Laplace=…, LTspice's); a coil as its blocked
    R (and L: Semi n = 1, LR-2) — each of those preceded by a "* loudpy {…}" line holding the part exactly, which
    from_spice reads back instead of the approximation.
    """
    import json
    lines = [f"* {title}", "* written by loudpy (lp.circuit.to_spice): '* loudpy' lines hold parts exactly"]

    def nm(prefix, name):
        return name if name[:1].upper() == prefix else f"{prefix}{name}"
    for p in circuit.parts:
        if isinstance(p, Resistor):
            lines.append(f"{nm('R', p.name)} {p.a} {p.b} {p.R:.12g}")
        elif isinstance(p, Capacitor):
            lines.append(f"{nm('C', p.name)} {p.a} {p.b} {p.C:.12g}")
        elif isinstance(p, Inductor):
            lines.append(f"{nm('L', p.name)} {p.a} {p.b} {p.L:.12g}")
        else:
            part = {"type": type(p).__name__, **{k: v for k, v in Circuit([p]).to_dict()["parts"][0].items()
                                                  if k != "type"}}
            lines.append("* loudpy " + json.dumps(part))
            if isinstance(p, Source):
                u = complex(p.U)
                node = p.a if not p.R_out else f"{p.name}_int"
                lines.append(f"{nm('V', p.name)} {node} {p.b} AC {abs(u):.12g} {np.degrees(np.angle(u)):.12g}")
                if p.R_out:
                    lines.append(f"R{p.name}_out {node} {p.a} {p.R_out:.12g}")
            elif isinstance(p, Filter):
                lines.append(f"{nm('E', p.name)} {p.out} 0 {p.inp} 0 Laplace={_laplace_text(p.H)}")
            elif isinstance(p, Coil):
                m = p.model
                if m is None:
                    raise ModelError(f"coil {p.name}: its motor is its driver's — give it first "
                                     "(circuit.with_motors({driver: motor}))")
                Le = getattr(m, "Le", 0.0) if getattr(m, "n", 1.0) == 1.0 else 0.0
                if Le:
                    lines += [f"R{p.name}_Re {p.a} {p.name}_n {m.Re:.12g}", f"L{p.name}_Le {p.name}_n {p.b} {Le:.12g}"]
                else:
                    lines.append(f"R{p.name}_Re {p.a} {p.b} {m.Re:.12g}")
            lines.append("* loudpy end")
    return "\n".join(lines + [".end", ""])


def from_spice(text: str) -> "Circuit":
    """
    A circuit from a SPICE netlist (LTspice's, ngspice's): R, L, C; V (its AC amplitude and phase, else its DC
    value) as the amplifier; E (a voltage-controlled source to ground: its gain, or LTspice's Laplace=…) as a filter
    block; "* loudpy {…}" lines (to_spice) as the parts they hold, exactly. Nodes GND / gnd are the ground "0".
    Dot commands are left out; other elements (diodes, transistors, subcircuits) are refused.
    """
    import json
    lines: list[str] = []
    for raw in str(text).splitlines():
        line = raw.strip()
        if line.startswith("+") and lines:
            lines[-1] += " " + line[1:].strip()
        elif line:
            lines.append(line)
    parts, skip, bad = [], False, []

    def node(n):
        return "0" if n.lower() == "gnd" else n
    for line in lines:
        low = line.lower()
        if low.startswith("* loudpy end"):
            skip = False
            continue
        if low.startswith("* loudpy {"):
            parts.extend(Circuit.from_dict({"parts": [json.loads(line[len("* loudpy "):])]}).parts)
            skip = True
            continue
        if skip or line.startswith(("*", ";", ".")):
            continue
        tok = line.split()
        kind, name = tok[0][0].upper(), tok[0]
        try:
            if kind in "RLC":
                v = spice_value(tok[3])
                cls = {"R": Resistor, "L": Inductor, "C": Capacitor}[kind]
                parts.append(cls(name, node(tok[1]), node(tok[2]), v))
            elif kind == "V":
                rest = [t.lower() for t in tok[3:]]
                if "ac" in rest:
                    i = rest.index("ac")
                    mag = spice_value(tok[3 + i + 1]) if len(tok) > 4 + i else 1.0
                    ph = spice_value(tok[3 + i + 2]) if len(tok) > 5 + i and not rest[i + 2].isalpha() else 0.0
                    u = mag * np.exp(1j * np.radians(ph))
                else:
                    vals = [t for t in tok[3:] if t.lower() != "dc"]
                    u = spice_value(vals[0]) if vals else 1.0
                parts.append(Source(name, node(tok[1]), node(tok[2]), U=complex(u)))
            elif kind == "E":
                if node(tok[2]) != "0" or node(tok[4]) != "0":
                    raise ModelError(f"{name}: a filter block reads and drives nodes to ground (E … out 0 in 0 …)")
                tail = " ".join(tok[5:])
                if tail.lower().startswith("laplace"):
                    H = _laplace(tail.split("=", 1)[1].strip().strip("{}"))
                else:
                    H = biquad([spice_value(tok[5])], [1.0])
                parts.append(Filter(name, node(tok[3]), node(tok[1]), H))
            else:
                bad.append(name)
        except (IndexError, ValueError) as e:
            raise ModelError(f"SPICE line {line!r}: {e}") from None
    if bad:
        raise ModelError(f"SPICE elements loudpy does not model: {bad} (R, L, C, V, E only)")
    return Circuit(*parts)


Circuit.to_spice = to_spice                  # type: ignore[attr-defined]
Circuit.from_spice = staticmethod(from_spice)  # type: ignore[attr-defined]
