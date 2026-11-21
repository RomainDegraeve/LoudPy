"""Common base of the studies: a meshed Model (or a Scene) in, Results out."""
from __future__ import annotations

from typing import Any


def finish(study, results):
    """results.model (the model description and its geometry files) and results.study (the settings)."""
    from loudpy.results.description import ModelDescription, describe, geometry_files, settings
    results.model = ModelDescription(describe(study.model), geometry_files(study.model))
    results.study = settings(study)
    return results

from loudpy.model.model import Model
from loudpy.errors import ModelError, StateError


class Study:
    """Base of the studies: the model (lp.Model, or lp.Scene for the studies that accept one), the results
    and the settings recorded with them."""
    accepts_scene = False

    def __init__(self, model: Model):
        from loudpy.scene.scene import Scene
        self.is_scene = isinstance(model, Scene)
        if not isinstance(model, Model) and not (self.is_scene and self.accepts_scene):
            kinds = "an lp.Model or an lp.Scene" if self.accepts_scene else "an lp.Model"
            raise ModelError(f"{type(self).__name__} takes {kinds}, got {type(model).__name__}")
        self.model = model
        self.results: Any = None

    def _parts(self) -> list:
        """The physics parts of the meshed model."""
        if self.model.meshed is None:
            raise StateError("mesh the model first: model.mesh(path)")
        return self.model.parts

    def _finish(self, results):
        """Attach what produced the results: the model description (with its geometry files) and the
        study settings (result format 3)."""
        return finish(self, results)

    def save(self, path, **metadata):
        """Write the last results (loudpy result format 2), with extra metadata (read back in .meta)."""
        if self.results is None:
            raise StateError("nothing to save: call solve() first")
        self.results.meta.update(metadata)
        return self.results.save(path)
