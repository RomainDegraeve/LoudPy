# COMSOL reference results

Drop COMSOL text exports here (Results ▸ Export ▸ Table / Point Graph / Data, text format, `%` headers) with the
file names of the cases in `tests/test_comsol.py::CASES`. Each case builds the same model in loudpy, reads the
file with `lp.read_comsol`, compares with `lp.compare` and records the numbers in the verification report. A case
whose file is missing is skipped (reported as such), so the harness runs before the reference results exist.

| file | COMSOL model to export | columns |
|---|---|---|
| `clamped_plate_modes.txt` | the clamped circular steel plate of `test_verification.py` (a = 0.1 m, h = 2 mm, 2D axisymmetric Solid Mechanics, eigenfrequencies of the axisymmetric bending modes) | mode number, eigenfrequency (Hz) |
| `speaker_front_spl.txt` | the HPNEW speaker of the tests, 1 N on the coil, front air with PML (Pressure Acoustics + Solid Mechanics, axisymmetric), point (0, 0.3) | freq (Hz), abs(p) (Pa) |
