import numpy as np


class DofMapMixin:
    """
    Provides `dof_maps()` for Studies that register their assemblers
    in `self._assemblers` (dict: field_name -> assembler instance).

    Returns one entry per field with:
        unique_tags   : (n_nodes,)  global node tags in assembler order
        dofs_per_node : int
        free_dofs     : (n_free,) or None  (None if BCs not applied / N/A)
        n_dof_full    : int   full (pre-BC) DOF count
    """

    def _require_assemblers(self):
        if not getattr(self, "_assemblers", None):
            raise RuntimeError(
                f"{type(self).__name__} has no registered assemblers. "
                "Run the assemble step before dof_maps()."
            )

    def dof_maps(self) -> dict:
        self._require_assemblers()
        out = {}
        for name, asm in self._assemblers.items():
            out[name] = {
                "unique_tags":   np.asarray(asm.unique_tags),
                "dofs_per_node": int(asm.dofs_per_node),
               
                "free_dofs":     getattr(asm, "_free", None),
               
                "n_dof_full":    int(asm.matrix_size),
            }
        return out
