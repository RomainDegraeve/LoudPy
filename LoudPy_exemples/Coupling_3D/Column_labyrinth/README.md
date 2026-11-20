# Coupling_3D/Column_labyrinth — three HPNEW drivers in a column with a folded transmission line

A column loudspeaker on a rigid ground: three identical drivers stacked on the front, loaded at the rear by a folded transmission line (labyrinth). It is the multi-driver version of `../Labyrinth`, built with `lp.Scene` (one `lp.Driver` placed three times).

- **Air:** one domain, made of the inside of the line, its mouth and the exterior half-space. The exterior uses 3D quadratic FE with the exact half-space DtN on a hemisphere.
- **Drivers:** the same Fourier modes (m = 0…3, cos and sin families) are used three times, each placed on its own axis. Their modal coordinates are stacked as top, middle, bottom. Each driver is wet on both faces by the same pressure field, and the drivers interact through it. No extra coupling term is needed.
- **Force:** 1 N on each voice coil, in phase (`DRIVE` in `Running_column.py`, complex amplitudes). The coil only drives m = 0. The m ≥ 1 motion of the drivers comes only from the non-uniform air load.
- **No stuffing.** Without damping the line resonances are sharp.

## Design (`column.py`, all dimensions in one place)
| | |
|---|---|
| cabinet (outside) | 17 × 67.5 × 22.5 cm (W × H × D), walls 15 mm, internal panels 12 mm |
| drivers | 3 × HPNEW on the front, axes 13 cm apart, the top one 8 cm below the inner top |
| line | 3 vertical channels, full inner width 14 cm: down (behind the drivers), up, down |
| channel depth | 8 → 6 → 4.5 cm, so the section goes 112 → 63 cm² |
| length | 1.99 m along the centre line, from the closed top of channel 1 |
| tuning | quarter-wave estimate 42 Hz |
| mouth | back wall, at ground level, 14 × 5 cm |

## How the scene is set up (`lp.Driver` + `lp.Scene`)
```python
hp = lp.Driver.from_modes(eig_path, [0, 1, 2, 3], f_max=6000, geometry=geo_path,
                          rear=REAR,     # lp.Profile(lp.BoundaryOf("subacou_rear"), start=…, end=…)
                          front=FRONT)   # lp.Profile(lp.Interface(SOLIDS, "subacou_front"), start=…)
scene = lp.Scene(cabinet, [hp.place(n, at=c) for n, c in zip(NAMES, CENTERS)])
scene.merged(ground="y", bands=BANDS, points_per_wavelength=4, max_size=0.07, driver_size=0.004)
scene.mesh(files)                                   # meshes built once, reused afterwards
res = lp.FrequencyResponse(scene, drive={"top": lp.Force(1), "middle": lp.Force(1),
                                         "bottom": lp.Force(1)}).solve(FREQS, store=files / "column.h5")
res = lp.load(files / "column.h5")                  # SceneResults
res.far_field(points); res.driver("top").excursion(); res.field("p", index=i)
```
`cabinet(occ)` (in `column.py`) is a builder: it adds the OCC solids and returns an `lp.Cabinet(inner, cuts, outer, openings)`. The air around the three drivers is carved by the scene; each wet surface is its own physical group, so every surface point is mapped to its own driver.

## Files
| file | does |
|---|---|
| `column.py` | Parameters, the folded-line geometry, and the line-length and tuning estimates. |
| `Running_column.py` | Coupled sweep, 20 Hz to `F_MAX`. Steps are 2 Hz below 300 Hz, then 10 Hz. Builds the air mesh of each band on the first run (`air_<f>Hz.msh`), then writes `column.h5` (loudpy result format 2; the pressure field is streamed to the file frequency by frequency). `python3 Running_column.py 40,100,300` runs a quick test into `column_test.h5`. |
| `Loading_column.py` | `column_response.pdf`: SPL at 1 m (front axis, 30° up and down, behind at the mouth), the excursion of each driver, vertical and horizontal directivity maps and polars, and the m ≥ 1 share and excursion phase of each driver. |
| `Plot_3D_column.py` | `column_views_3d.pdf`: the column cut through the driver axes, showing the line in section. Shows SPL inside and outside with iso-lines, and the three deformed drivers at one instant and one scale. |

## Running
```bash
python3 LoudPy_exemples/Eigen_study/Fourier/Running_script.py
python3 LoudPy_exemples/Coupling_3D/Column_labyrinth/Running_column.py
python3 LoudPy_exemples/Coupling_3D/Column_labyrinth/Loading_column.py
python3 LoudPy_exemples/Coupling_3D/Column_labyrinth/Plot_3D_column.py
```

- **First script:** computes the driver modes. Skip it if `eigen_fourier.h5` already exists.
- **Memory:** the default `F_MAX` is 1 kHz. Its mesh has about 100k nodes and peaks near 2 GB. The 2 kHz mesh (169k nodes) is built too, but needs more than 8 GB of RAM. Set `F_MAX = 2000` on a larger machine.
- **Size of the structural block:** 3 × 320 driver modes (`DRV_F_MAX` = 6 kHz).

## Results (default settings)
The sweep covers 211 frequencies, 20–1000 Hz, in 24 min on an M2 (8 GB). The mesh has 98.5k nodes; each driver's wet surface is 281.8 cm², front + rear.

- **Line tuning:** the excursion minimum of all three drivers is at 38 Hz, against the 42 Hz quarter-wave estimate. There the mouth carries the radiation (`column_views_3d.pdf`, 40 Hz page), and the line's first peak is at 32 Hz.
- **Drivers are not loaded equally:** the line couples them, so the same force gives different motions.
  - Near the main resonance (106 Hz), the top driver moves 1.5× and the bottom one 0.9× as much as the middle driver, and the bottom driver's peak moves up to 112 Hz.
  - Around 110–130 Hz the top and bottom drivers swing by up to ±90–170° in phase relative to the middle driver.
  - Below about 90 Hz and above about 300 Hz the three move together.
- **Rocking from the air load alone:** at the notches of the excursion (130, 226, 490 Hz), the m ≥ 1 orders carry up to 44 % (middle driver), 24 % (bottom) and 14 % (top) of the kinetic energy. The coil drives only m = 0, so this comes entirely from the non-uniform pressure along the column. With a single axisymmetric model this motion would be missing.
- **Column directivity:** horizontally the column stays nearly omnidirectional up to about 150 Hz. Vertically, from about 250 Hz the three sources interfere: SPL 30° up has a 40 dB notch at 270 Hz, and above 400 Hz the vertical lobing is strong (directivity maps).
- **No stuffing:** the line resonances (32, 106, 152, 230 Hz…) are sharp. Damping the line (equivalent fluid) is the natural next step.
