"""Common base of the studies: a meshed Model in, Results out."""
from __future__ import annotations

from loudpy.model.model import Model


class Study:
    def __init__(self, model: Model):
        if not isinstance(model, Model):
            raise TypeError(f"{type(self).__name__} takes an lp.Model, got {type(model).__name__}")
        self.model = model
        self.results = None

    def _problem(self):
        if self.model.meshed is None:
            raise RuntimeError("mesh the model first: model.mesh(path)")
        return self.model.as_problem()

    def save(self, path, **metadata):
        """Write the last results (loudpy result format 2), with extra metadata (read back in .meta)."""
        if self.results is None:
            raise RuntimeError("nothing to save: call solve() first")
        self.results.meta.update(metadata)
        return self.results.save(path)
