from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DenoisingStrengthThresholds:
    r_D_hom_min: float = 0.65
    r_D_hom_max: float = 0.85
    r_D_hom_identity: float = 0.90
    r_D_hom_oversmooth: float = 0.60
    e_D_min: float = 0.10
    e_D_identity: float = 0.05
    eta_res_edge_max: float = 0.10
    c_lesion_min: float = 0.90
    A_NPS_min: float = 0.50
    A_NPS_max: float = 0.85
    d_shape_max: float = 0.15


class DenoisingStrengthController:
    def __init__(self, thresholds: DenoisingStrengthThresholds | None = None):
        self.thresholds = thresholds or DenoisingStrengthThresholds()

    @classmethod
    def from_mapping(cls, mapping: Dict[str, float]) -> "DenoisingStrengthController":
        values = DenoisingStrengthThresholds()
        for key, value in mapping.items():
            if hasattr(values, key):
                setattr(values, key, float(value))
        return cls(values)

    def evaluate(self, metrics: Dict[str, float]) -> Dict[str, object]:
        t = self.thresholds
        r = float(metrics.get("r_D_hom_mean", 1.0))
        e = float(metrics.get("e_D_mean", 0.0))
        eta = float(metrics.get("eta_res_edge_mean", 1.0))
        lesion = float(metrics.get("c_lesion_mean", 1.0))
        amp = float(metrics.get("A_NPS_mean", 1.0))
        shape = float(metrics.get("d_shape_mean", 1.0))

        identity = r > t.r_D_hom_identity or e < t.e_D_identity
        over_smoothing = r < t.r_D_hom_oversmooth or lesion < t.c_lesion_min
        structure_deletion = eta > t.eta_res_edge_max
        release = (
            t.r_D_hom_min <= r <= t.r_D_hom_max
            and e >= t.e_D_min
            and eta <= t.eta_res_edge_max
            and lesion >= t.c_lesion_min
            and t.A_NPS_min <= amp <= t.A_NPS_max
            and shape <= t.d_shape_max
            and not identity
            and not over_smoothing
            and not structure_deletion
        )

        reasons: List[str] = []
        if identity:
            reasons.append("conservative_identity_collapse")
        if over_smoothing:
            reasons.append("over_smoothing")
        if structure_deletion:
            reasons.append("structure_deletion")
        if not (t.r_D_hom_min <= r <= t.r_D_hom_max):
            reasons.append(f"r_D_hom_out_of_range:{r:.4f}")
        if e < t.e_D_min:
            reasons.append(f"e_D_below_min:{e:.4f}")
        if not (t.A_NPS_min <= amp <= t.A_NPS_max):
            reasons.append(f"A_NPS_out_of_range:{amp:.4f}")
        if shape > t.d_shape_max:
            reasons.append(f"d_shape_above_max:{shape:.4f}")

        return {
            "flags": {
                "identity_collapse": bool(identity),
                "over_smoothing": bool(over_smoothing),
                "structure_deletion": bool(structure_deletion),
                "release_D": bool(release),
            },
            "reasons": reasons,
            "recommended_action": "release" if release else "hold_or_rollback",
        }

    def thresholds_dict(self) -> Dict[str, float]:
        return self.thresholds.__dict__.copy()
