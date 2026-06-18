# loudpy

A fem python API.

## Installation

### PDM Installation

loudpy relies on the Python Development Master (PDM) for package management. 



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
