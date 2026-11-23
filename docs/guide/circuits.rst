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

**How it is solved.** By modified nodal analysis: the node voltages and the currents of the inductors, sources,
filters and coils are unknowns solved at each frequency with the drivers' modes, the air and the walls — nothing is
iterated. A coil's row is ``Z(s) i + Bl v_coil − (v_a − v_b) = 0`` (its back-EMF through its driver's coil
velocity) and its current pushes its driver with ``Bl i``. A circuit made of a source and one coil is exactly
``lp.Voltage``.

**A 2D model** is driven the same way (``lp.FrequencyResponse(model, drive=circuit)``, a coupled model): the
circuit holds one coil, on the model's ``lp.Coil`` boundary (its ``driver`` left empty); the results keep the coil's
current and voltage per frequency (``res.impedance()``) and the whole circuit (``res.circuit``).

**Alone.** ``circuit.solve(freqs)`` solves the circuit with its coils blocked (no back-EMF): the filters'
transfer functions, the loads; ``Circuit.to_dict`` / ``from_dict`` give it as plain data (the results keep it).
