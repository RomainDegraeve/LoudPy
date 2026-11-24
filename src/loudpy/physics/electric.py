"""
The electrical side of a frequency response (lp.Voltage, lp.Circuit), solved on top of the structure and the air.

The structure and the air are linear: they are solved once per coil, for a unit force on it — its unit response X_k
(every field, the modal amplitudes, the pressures). The coils meet them only through the mobility matrix Y (n_f, K, K),
the velocity of coil j for a unit force on coil k (the air, the walls and the other drivers included). The circuit is
then solved alone at each frequency (modified nodal analysis, loudpy.circuit), each coil's back-EMF in its branch:

    Z_j(s) i_j − (v_a − v_b) + Bl_j Σ_k Y_jk Bl_k i_k = 0

and the coil forces w_k = Bl_k i_k weight the unit responses: the driven response is Σ_k w_k X_k. This is the
coupled solution exactly, and the circuit (filters, crossover, coil values) can change after the study:
Results.redrive.

    circuit = as_circuit(drive, coils)        an lp.Voltage (or {driver: Voltage}) as a circuit; a Force: None
    W, E = solve(circuit, freqs, Y, coils)    coil forces (n_f, K) and the circuit's unknowns (n_f, size)
    W = force_weights(drive, coils, n_f)      a Force: its amplitudes on the coils
    coil_scalars(circuit, E, freqs)           a 2D model's coil: I_coil, U, Ze, Bl (Re, Le, n of a Semi coil)

coils: the drivers in the order of the unit responses (a scene: their names; a 2D model: [None], its one coil).
"""
from __future__ import annotations

import numpy as np

from loudpy.errors import ModelError


def as_circuit(drive, coils: list):
    """The circuit of an electrical drive: an lp.Circuit itself; an lp.Voltage (every coil) or {driver: lp.Voltage}
    as one amplifier and one coil (Semi: Re, Bl, Le, n) per driver; None for a Force (or {driver: Force})."""
    from loudpy.circuit import Circuit, Coil, Semi, Source
    from loudpy.model.drives import Force, Voltage
    if isinstance(drive, Circuit):
        return drive
    per = drive if isinstance(drive, dict) else {c: drive for c in coils}
    values = list(per.values())
    if all(isinstance(v, Force) for v in values):
        return None
    if not all(isinstance(v, Voltage) for v in values):
        raise ModelError("drivers are all driven by a Force or all by a Voltage (or by one lp.Circuit)")
    unknown = sorted(set(per) - set(coils), key=str)
    if unknown:
        raise ModelError(f"drive: no driver {unknown} (drivers: {coils})")
    missing = [c for c in coils if c not in per]
    if missing:
        raise ModelError(f"a Voltage drive names every driver (each has its own coil: Re, Bl, Le): {coils}")
    parts = []
    for c in coils:
        v = per[c]
        node, tag = (f"in_{c}", f"_{c}") if c is not None else ("in", "")
        parts += [Source(f"amp{tag}", node, U=complex(v.amplitude)),
                  Coil(f"coil{tag}", node, "0", Semi(Re=float(v.Re), Bl=float(v.Bl), Le=float(v.Le or 0.0),
                                                     n=1.0 if v.n is None else float(v.n)), driver=c)]
    return Circuit(*parts)


def columns(circuit, coils: list) -> list[int]:
    """For each coil of the circuit, the unit response it drives: its driver's place in `coils` (a 2D model: its
    one coil)."""
    if coils == [None]:
        if len(circuit.coils) != 1:
            raise ModelError(f"a 2D model has one coil: its circuit holds one Coil part ({len(circuit.coils)} given)")
        return [0]
    out = []
    for c in circuit.coils:
        if c.driver is None:
            raise ModelError(f"coil {c.name}: name the driver it belongs to (driver=…; drivers: {coils})")
        if c.driver not in coils:
            raise ModelError(f"coil {c.name}: driver {c.driver!r} is not one of {coils} (a result keeps the drivers its "
                             "study drove: drive another one by solving again)")
        out.append(coils.index(c.driver))
    twice = sorted({coils[k] for k in out if out.count(k) > 1})
    if twice:
        raise ModelError(f"driver(s) {twice} have several coils in the circuit: one coil per driver")
    return out


def solve(circuit, freqs, Y, coils: list) -> tuple[np.ndarray, np.ndarray]:
    """The circuit at every frequency with its coils moving (mobility Y (n_f, K, K) of the unit responses, in the order
    of `coils`): (W (n_f, K) the coil forces Bl·i on each unit response — 0 for a driver without a coil, E (n_f, size)
    the circuit's unknowns)."""
    freqs = np.atleast_1d(np.asarray(freqs, float))
    circuit.check(freqs)
    cols = columns(circuit, coils)
    Y = np.asarray(Y, complex).reshape(len(freqs), len(coils), len(coils))
    rows = [circuit.index(c.name) for c in circuit.coils]
    Bl = np.array([float(c.model.Bl) for c in circuit.coils])
    b = circuit.rhs()
    W = np.zeros((len(freqs), len(coils)), complex)
    E = np.zeros((len(freqs), circuit.size), complex)
    for i, f in enumerate(freqs):
        A = circuit.matrix(2j * np.pi * f)
        A[np.ix_(rows, rows)] += Bl[:, None] * Y[i][np.ix_(cols, cols)] * Bl[None, :]   # the back-EMFs
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x = np.full(len(b), np.nan)
        if not np.all(np.isfinite(x)):
            raise ModelError(f"the circuit with its coils moving has no single solution at {f:.4g} Hz")
        E[i] = x
        W[i, cols] = Bl * x[rows]
    return W, E


def force_weights(drive, coils: list, n_f: int) -> np.ndarray:
    """A Force drive's amplitudes on the unit responses (n_f, K): every coil (lp.Force) or {driver: lp.Force}."""
    from loudpy.model.drives import Force
    if isinstance(drive, Force):
        amps = [complex(drive.amplitude)] * len(coils)
    else:
        amps = [complex(drive[c].amplitude) if c in drive else 0.0 for c in coils]
    return np.tile(np.asarray(amps, complex), (n_f, 1))


def coil_scalars(circuit, E, freqs) -> dict:
    """Per-frequency values of a 2D model's one coil: I_coil, U (its voltage), Ze (blocked), Bl; Re, Le, n of a
    semi-inductance coil (Thiele–Small)."""
    from loudpy.circuit import GROUND, Semi
    coil = circuit.coils[0]
    E = np.asarray(E)
    s = 2j * np.pi * np.asarray(freqs, float)

    def v(node):
        return np.zeros(len(E), complex) if node == GROUND else E[:, circuit.index(node)]
    m = coil.model
    out = dict(I_coil=E[:, circuit.index(coil.name)], U=v(coil.a) - v(coil.b),
               Ze=np.array([complex(m.impedance(z)) for z in s]), Bl=np.full(len(E), float(m.Bl)))
    if isinstance(m, Semi):
        out.update(Re=np.full(len(E), float(m.Re)), Le=np.full(len(E), float(m.Le)), n=np.full(len(E), float(m.n)))
    return out
