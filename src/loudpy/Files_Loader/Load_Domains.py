from enum import Enum

class Domain(str, Enum):
    """Physics-domain identifier — single source of truth."""
    ACOU = "acou"
    MECA = "meca"

    @property
    def field_name(self) -> str:
        return {"acou": "p_acou", "meca": "u_meca"}[self.value]

    @property
    def default_dpn(self) -> int:
        return {"acou": 1, "meca": 2}[self.value]

    @property
    def subgroup_prefixes(self) -> tuple[str, ...]:
        return {"acou": ("subdomainacou",),
                "meca": ("subdomainmeca",)}[self.value]

    @classmethod
    def coerce(cls, v: "Domain | str") -> "Domain":
        return v if isinstance(v, cls) else cls(v)

