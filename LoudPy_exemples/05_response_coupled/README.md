# 05_response_coupled — coupled structure + air response, 20 Hz – 20 kHz

The driver radiating into air truncated by a PML, 1 N on the coil. The air and PML meshes follow the wavelength
(`lp.PerWavelength`): one mesh per band, built on the first run.

```bash
python3 LoudPy_exemples/05_response_coupled/run.py    # ~2 min → Results/Files/sweep.h5 (+ the band meshes)
python3 LoudPy_exemples/05_response_coupled/plot.py   # SPL, directivity maps and polars, pressure fields
```

**Shows**: `lp.Air`, `lp.PML`, `lp.FluidStructure`, remeshing per wavelength, SPL on `lp.Arc`s (frequency × angle
maps), `lp.plot.directivity`, pressure and displacement fields, CSV export of a directivity map.
