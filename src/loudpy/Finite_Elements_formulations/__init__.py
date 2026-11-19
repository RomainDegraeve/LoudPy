"""Moved to loudpy.fem (kept so that existing imports keep working)."""
from loudpy.fem.elements.tri6_axi import (compute_H_T6, compute_Q_T6, compute_H_T6_pml,  # noqa: F401
                                         compute_Q_T6_pml, compute_M_T6, compute_K_T6, compute_FSI_T6,
                                         compute_unit_force_T6, compute_element_TL)
