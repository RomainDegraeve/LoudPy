# 03_coupled_modes_radiation — coupled structure + air modes, radiating

The driver with the air in front and behind it, radiating into free space through an **exact** radiation
condition (Dirichlet-to-Neumann on a sphere, `lp.Radiation`): the coupled modes are complex (radiation damping).

```bash
python3 examples/03_coupled_modes_radiation/run.py    # ~7 min → Results/Files/modes.h5
python3 examples/03_coupled_modes_radiation/plot.py   # eigenvalues, structural / acoustic modes
```

**Shows**: `lp.Air`, `lp.FluidStructure`, `lp.Radiation`, `lp.Modes(model, physics="coupled")`, the coupled mode
sets and their classification against the in-vacuo modes of `01_structure_modes`.
**Result**: pistonic mode 112 Hz with the air (115 Hz in vacuo), radiation damping of the coupled modes.
