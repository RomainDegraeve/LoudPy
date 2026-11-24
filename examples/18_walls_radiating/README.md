# 18 — Elastic walls in a radiating labyrinth

The HPNEW driver (modes of `02_azimuthal_modes`) in a cabinet drawn as solids in a STEP file (`cabinet.py` writes
`cabinet.step`): six panels, two labyrinth boards folding the path from the driver down to a slot at the bottom of
the front, and the slot's air. Every panel and board is an elastic shell of its own measured thickness (the back in
plywood); the air is one domain — the labyrinth, the slot and the outside on a rigid ground (FE + exact half-space
DtN) — and every wet face of the walls moves with its shell: the boards on both faces, the panels inside and out.

```
cabinet.py                 the cabinet's solids → cabinet.step (0 bottom, 1 left, 2 right, 3 top, 4 back, 5 front,
                           6 upper board, 7 lower board, 8 the slot's air)
run.py                     the scene; python3 run.py meshes, computes the wall modes and the response with elastic
                           and with rigid walls (Results/Files/)
plot.py                    SPL 1 m ahead, coil excursion, walls / driver volume velocity, RMS normal displacement
                           of each panel, a 3D view
18_walls_radiating.loudpy.json   the same simulation for the interface (python -m loudpy.ui on this folder)
```

```python
cabinet = lp.cabinets.Step("cabinet.step", openings=[OPENING])     # a point of the slot's air
scene = lp.Scene.vented_box(cabinet, driver(), at=(X0, DRIVER_Y), f_max=1000)   # one merged air domain
scene.walls("MDF", thickness=None,                                 # each panel's thickness measured on its solid
            panels={4: lp.cabinets.Panel("Plywood")})
```

Panels found on the inside air (checked, not run): bottom 18, left / right / top 15, back 22 mm (plywood), front
18 mm, upper and lower boards 12 mm, all measured on their solids; inside air 10.33 L; the first wall mode 1.53 kHz
(panel edges clamped), above the 1 kHz band of the response.

The front and the driver: the driver's model (02) is an axisymmetric 2D model whose air meets the cabinet's on two
planes 1 mm apart, its flange: the rear baffle plane Z_B = 31 mm and the front face Z_F = 32 mm. The front panel is
18 mm thick up to Z_F; behind Z_B it has the driver's cut-out (a hole of 58 mm radius, the basket's room), and loudpy
cuts the driver itself in the last millimetre. To try 15 mm: FRONT = 0.015 in cabinet.py (python3 cabinet.py writes
the file again).

How the walls are coupled (docs: guide/shells): one shell per panel on its inner face (a board: on one side); every
other wet face (the outer face, the board's other side) moves with the shell at its nearest point — rigid through
the thickness (thin walls). Checked in the tests on a smaller box: each wall mode displaces the air on its two faces
by opposite volumes (−0.97 … −1.00), walls 100 × stiffer bring the response 100 × closer to rigid walls, and the
power put in at the coil is what the modes dissipate plus what the hemisphere radiates (to 1e-10).

Not run yet: the two responses take about an hour (estimate).

## Electrical side

The driver has its motor (`shared/enclosure.driver`, and the document: Semi, Re 8 Ω, Bl 7 T·m, Le 4 mH, n 0.8) and
the simulation holds the circuit `amp` (a 2.83 V amplifier on the coil; Electrical in the interface). The studies are
driven by 1 N; with one driver the result keeps the coil's unit response, so it can be driven by `amp` — or any circuit
drawn in the interface — after the study, without solving again (`res.redrive(circuit)`).
