Circuits: amplifier, filters, coils
===================================

The electrical side of a loudspeaker — the amplifier, active or DSP filters, passive parts and the drivers' coils —
is an :class:`lp.Circuit <loudpy.circuit.Circuit>` of parts on named nodes (``"0"`` is the ground). It drives a
scene like a force or a voltage does:

.. code-block:: python

   import loudpy as lp
   from loudpy.circuit import Source, Filter, Resistor, Coil, LR2, linkwitz_riley, peq

   circuit = lp.Circuit(
       Source("amp", "in", U=2.83, R_out=0.05),                       # the amplifier
       Filter("xo", "in", "w", linkwitz_riley(4, 2000.0, "low") * peq(80.0, 3.0, 1.0)),
       Resistor("Rs", "w", "c", 0.5),
       Coil("main", "c", "0", LR2(Re=5.6, Bl=6.1, Le=3e-4, R2=2.0, L2=1e-4), driver="main"),
   )
   res = lp.FrequencyResponse(scene, drive=circuit).solve(freqs)
   res.input_impedance(), res.impedance("main"), res.voltage("c"), res.current("Rs"), res.transfer("w")

**Parts.** ``Resistor``, ``Capacitor``, ``Inductor``; ``Source`` (the amplifier: a voltage behind its output
resistance); ``Filter`` (an ideal block, ``v(out) = H(s) v(inp)``, whatever its load); ``Coil`` (a driver's coil,
bound to the scene's driver by name).

**Filter blocks** (``loudpy.circuit``): ``butterworth(order, fc, "low" | "high")``, ``linkwitz_riley(order, fc, …)``
(even orders; the order-4 low and high pass sum flat), ``peq(f0, gain_db, q)``, ``low_shelf`` / ``high_shelf``,
``gain(db)``, ``delay(t)``, ``biquad(num, den)``; blocks cascade by multiplication.

**Coil models** (the blocked coil and its force factor ``Bl``): ``Semi`` (Re + sⁿLe), ``Leach`` (Re + K sⁿ),
``LR2`` (Re + sLe + R2 ∥ sL2), ``Wright`` (Re + Krm ωᴱʳᵐ + j Kxm ωᴱˣᵐ).

**The motor belongs to the driver.** A driver's coil model is its *motor*, given once on its 2D model's coil item —
``lp.Coil("coil_drive", motor=Semi(Re=5.6, Bl=6.1, Le=3e-4))`` — and kept in its result files: a scene's driver
made by ``lp.Driver.from_modes`` reads it from its modes file (``motor=`` there overrides it). A circuit's ``Coil``
then only names its driver — ``Coil("main", "c", "0", driver="main")`` — and ``lp.Voltage(2.83)`` needs only its
amplitude; a coil model given in the circuit (``model=``) or values in the voltage (``Re``, ``Bl``, ``Le``, ``n``)
override the motor. Results keep the motors (``res.motors``), so ``res.redrive`` drives them too.

**How it is solved.** The structure and the air are linear: each driven coil is solved once for a unit force on it
(its *unit response*: the modal amplitudes, the pressures, the field), with the drivers' modes, the air and the
walls. The coils meet them only through the mobility matrix ``Y`` — the velocity of coil ``j`` for a unit force on
coil ``k``, the air and the other drivers included. The circuit is then solved at each frequency by modified nodal
analysis (the node voltages and the currents of the inductors, sources, filters and coils its unknowns), a coil's
row ``Z(s) i_j − (v_a − v_b) + Bl_j Σ_k Y_jk Bl_k i_k = 0`` holding its back-EMF, and the coil forces ``Bl i``
weight the unit responses. This is the coupled solution exactly (nothing is iterated). A circuit made of a source
and one coil is exactly ``lp.Voltage``, which is solved the same way.

**Changing the circuit after the study.** Since the results keep the unit responses, their drive changes without
solving again — another crossover, an EQ, other coil values, a voltage for a force:

.. code-block:: python

   res = lp.FrequencyResponse(scene, drive=circuit).solve(freqs, store="a.h5")
   res2 = res.redrive(circuit2)               # instant: the field, SPL, impedance with circuit2
   res2.spl(at=(0.1, 0.2, 1.0)), res2.impedance("main"), res2.save("b.h5")

A result is redrivable (``res.redrivable``) when its study was driven by a voltage or a circuit, or by a force on
one coil; one force on several drivers at once is solved as one response.

**What a circuit must be.** ``circuit.check()`` (done before any solve) refuses, saying why: no source; a node joined
to one part only; two parts setting the voltage of one node — sources and filter outputs are ideal, each fixes its
node's voltage, so two filter outputs on one node contradict each other (chain their stages in one ``Filter``, or
join the outputs through resistors); a node with no path to the ground through parts carrying a current (a filter's
input draws none); equations without a single solution.

**A 2D model** is driven the same way (``lp.FrequencyResponse(model, drive=circuit)``, with or without air): the
circuit holds one coil, on the model's ``lp.Coil`` boundary (its ``driver`` left empty); the results keep the coil's
current and voltage per frequency (``res.impedance()``) and the whole circuit (``res.circuit``).

**Alone.** ``circuit.solve(freqs)`` solves the circuit with its coils blocked (no back-EMF): the filters'
transfer functions, the loads; ``Circuit.to_dict`` / ``from_dict`` give it as plain data (the results keep it).

**SPICE netlists.** ``circuit.to_spice(title)`` writes a netlist LTspice reads (the amplifier as ``V … AC``, a filter
block as LTspice's ``E … Laplace=…``, a coil as its blocked R and L), each part SPICE cannot hold exactly preceded by
a ``* loudpy {…}`` line; ``lp.Circuit.from_spice(text)`` reads such a netlist back exactly, and plain SPICE ones:
R, L, C, V (its AC amplitude and phase), E (a gain or ``Laplace=``), scale suffixes (``4.7k``, ``10uF``, ``2Meg``)
and ``GND``. Coils are added in loudpy (a netlist has no ``Bl``).

**In documents and scripts.** A circuit is a node of the simulation document (``lp.schema``: ``Circuit`` and the
``circuit.*`` blocks; a filter's stages kept as they were made), so a saved study keeps its circuit and its script
writes it as ``lp.Circuit(lp.circuit.Source(…), …)``.
