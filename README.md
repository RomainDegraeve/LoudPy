# LoudPy


LoudPy is an open-source Python API for finite element modelling of axisymmetric loudspeakers, 
released under the [GPL-3.0 license](https://www.gnu.org/licenses/gpl-3.0.html).

## Overview

LoudPy provides a coupled FEM solver dedicated to loudspeaker design, featuring harmonic 
frequency-domain, time-domain nonlinear, and complex eigenvalue solvers.

It aims to bridge the gap between non-user-friendly general-purpose open-source FEM solvers 
and costly user-friendly commercial FEM software dedicated to loudspeaker design.

It handles 2D axisymmetric geometries and solves the fully coupled electro-mechanical-acoustic 
problem. Geometry is imported from STEP files, which are converted into BREP format. Named 
entities are then assigned to subdomains and boundaries to define material properties, coupling 
interfaces, and boundary conditions.

The results are exported in .H5 format, solve field and coordinates are directly solved as numpy arrays.

This project was developed by Romain Degraeve as part of the 
[IMDEA Master's programme in Acoustics](https://iags.univ-lemans.fr/en/education-programs/master-s-degrees-in-acoustics/parcours-en-anglais/imdea.html#Generalinformation1-1)
(International Master Degree in Acoustics).

### Key features

- 2D axisymmetric FEM for structural and acoustic domains (frequency-domain harmonic solver)
- Complex eigenfrequency analysis for mechanical domain
- Nonlinear time-domain solver for mechanical domain
- Rayleigh and hysteretic damping models (both support non-proportional formulations)
- Perfectly Matched Layer (PML) for acoustic radiation
- Mesh generation via the Gmsh Python API
- Built on NumPy and SciPy only — no external FEM library

The `examples/` folder contains a few application examples of LoudPy.

## Contributing

If you want to contribute to this open-source project, feel free to reach out:

- Email: [rom2graeve@gmail.com](mailto:rom2graeve@gmail.com)
- LinkedIn: [Romain Degraeve](https://www.linkedin.com/in/romain-degraeve-558073290/)

## Publication

LoudPy will be presented at Acousticum 2026 [paper link](https://conforg.fr/bin/v2/pdfdoc?dir=fa2026&ref=226&pwd=AGGBVU) 
(ref. 226).

## Installation
### PIP Installation
The easiet way to install LouPy is to run the following command in your terminal.

```bash
pip install loudpy
```


### PDM Installation

Loudpy relies on the Python Development Master (PDM) for package management. 




#### UV Installation

UV is a modern Python package manager that provides a simple and efficient way to manage Python packages and environments. For example, different python intepreter can be installed through the commands:

```bash
uv python install 3.11 3.12 3.13
```

ThisPython API is developped under python 3.13, so with uv run:

```bash
uv python install 3.13
```

Then, to install PDM using UV, run the following command:

```bash
uv pip install pdm
```

#### PIPX Installation

To install PDM using pipx, run the following command:

```bash
pipx install pdm
```

Prefer `pipx` as it allows you to install Python applications in isolated environments, preventing conflicts between packages.

#### Verify PDM Installation

When the installation is complete, you can verify that PDM is installed correctly by running:

```bash
pdm --version
```

### loudpy Installation

To install loudpy for development purposes, use PDM to install it in editable mode. This allows you to make changes to the source code and have those changes reflected immediately without needing to reinstall the package. To do this, navigate to the directory where you have the loudpy source code and run:

```bash
pdm install -e .
```
