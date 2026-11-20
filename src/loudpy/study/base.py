"""Common base of the studies: a meshed Model (or a Scene) in, Results out."""
from __future__ import annotations

from loudpy.model.model import Model


class Study:
    accepts_scene = False

    def __init__(self, model: Model):
        from loudpy.scene.scene import Scene
        self.is_scene = isinstance(model, Scene)
        if not isinstance(model, Model) and not (self.is_scene and self.accepts_scene):
            kinds = "an lp.Model or an lp.Scene" if self.accepts_scene else "an lp.Model"
            raise TypeError(f"{type(self).__name__} takes {kinds}, got {type(model).__name__}")
        self.model = model
        self.results = None

    def _parts(self) -> list:
        """The physics parts of the meshed model."""
        if self.model.meshed is None:
            raise RuntimeError("mesh the model first: model.mesh(path)")
        return self.model.parts

    def save(self, path, **metadata):
        """Write the last results (loudpy result format 2), with extra metadata (read back in .meta)."""
        if self.results is None:
            raise RuntimeError("nothing to save: call solve() first")
        self.results.meta.update(metadata)
        return self.results.save(path)
