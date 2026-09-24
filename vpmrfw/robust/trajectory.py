from __future__ import annotations
from dataclasses import dataclass
import numpy as np


class CertificationFailure(RuntimeError):
    """Raised when a theorem-certified execution cannot justify its next step."""


@dataclass(frozen=True)
class FullCoverTrajectoryGuard:
    """Guard for an independently verified chart cover of the whole outer polytope.

    This object deliberately does *not* infer certification from floating-point condition
    numbers.  It may only be constructed with ``verified=True`` when the user has an
    external/interval/rational verification matching the constants in ``RegularityCaps``.
    """
    verified: bool = False
    provenance: str = ""

    def accept_segment(self, x: np.ndarray, x_plus: np.ndarray) -> bool:
        del x, x_plus
        return bool(self.verified)
