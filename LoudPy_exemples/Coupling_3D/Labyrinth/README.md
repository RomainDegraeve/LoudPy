# Coupling_3D/Labyrinth — HPNEW in a folded transmission-line cabinet on a rigid ground

A labyrinth (folded quarter-wave transmission line) is simulated as one air domain, the same way as the vented box in `../Driver_in_cavity`.

- **Air:** the inside of the line, its mouth and the exterior half-space. The exterior uses 3D quadratic FE with the exact half-space DtN on a hemisphere.
- **Driver:** its Fourier modes (m = 0…3) are strongly coupled to that air on both faces. The rear of the cone feeds the closed end of the line, and the mouth radiates into the room.
- **No stuffing yet.** Without damping the line resonances are sharp. The equivalent-fluid model (Delany–Bazley–Miki) is the next step.

## Design (`labyrinth.py`, all dimensions in one place)
| | |
|---|---|
| cabinet (outside) | 19 × 41 × 30 cm (W × H × D), walls 15 mm, internal panels 12 mm |
| line | 4 layers folded by 3 panels, full inner width 16 cm |
| line height | h = 14 → 10 → 7.2 → 5 cm, so the section goes 224 → 80 cm² (taper 2.8:1) |
| length | 1.20 m along the centre line |
| tuning | quarter-wave estimate 70 Hz; the simulated excursion minimum is near 60 Hz |
| driver | front, top, firing into the closed end of the line |
| mouth | front, at ground level, 16 × 5 cm |

## Files
| file | does |
|---|---|
| `labyrinth.py` | Parameters, the folded-line geometry, and the line-length and tuning estimates. |
| `Build_labyrinth.py` | Exports the cabinet (`labyrinth_cabinet.step` / `.geo_unrolled`) and the line air (`labyrinth_line.step`). Writes one merged air mesh per frequency band (`labyrinth_<f>Hz.msh`) plus `labyrinth.json`. |
| `Running_labyrinth.py` | Coupled sweep, 20 Hz to `F_MAX`. Steps are 2 Hz below 300 Hz, then 10 Hz. Writes `labyrinth.npz` (saved after every band) and `labyrinth_field.h5` (flushed after every frequency). `python3 Running_labyrinth.py 40,70,100` runs a quick test into `labyrinth_test.*`. |
| `Loading_labyrinth.py` | `labyrinth_response.pdf`: SPL at 1 m (driver axis, mouth, off axis) and excursion, against the sealed and vented boxes when their results exist; directivity maps; polar curves. |
| `Plot_3D_labyrinth.py` | `labyrinth_views_3d.pdf`: 3D scenes cut through the driver axis, showing the whole folded line in section. Includes SPL with iso-lines and the deformed driver. |

## Running
```bash
python3 LoudPy_exemples/Eigen_study/Fourier/Running_script.py
python3 LoudPy_exemples/Coupling_3D/Labyrinth/Build_labyrinth.py
python3 LoudPy_exemples/Coupling_3D/Labyrinth/Running_labyrinth.py
python3 LoudPy_exemples/Coupling_3D/Labyrinth/Loading_labyrinth.py
python3 LoudPy_exemples/Coupling_3D/Labyrinth/Plot_3D_labyrinth.py
```

- **First script:** computes the driver modes. Skip it if `eigen_fourier.h5` already exists.
- **Memory:** the default `F_MAX` is 2 kHz, which needs meshes of about 63k and 84k nodes. The 3 kHz mesh (147k nodes) is built too, but needs more than 8 GB of RAM to run without swapping. Set `F_MAX = 3000` on a larger machine.
- **Run time:** about 25 minutes on an M2.
