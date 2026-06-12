from typing import Tuple, Optional
import math
from trace_ct.config.schema import AlignmentConfig

def verify_physical_alignment(
    reg_spacing: Optional[Tuple[float, ...]],
    reg_origin: Optional[Tuple[float, ...]],
    reg_shape: Tuple[int, ...],
    hr_spacing: Optional[Tuple[float, ...]],
    hr_origin: Optional[Tuple[float, ...]],
    hr_shape: Tuple[int, ...],
    config: AlignmentConfig
) -> Tuple[bool, str]:
    """
    Verifies physical coordinate alignment between REG and HR volumes.
    Returns (True, "OK") if valid, otherwise (False, ErrorMessage).
    """
    if config.require_physical_alignment:
        if not reg_spacing or not hr_spacing:
            return False, "Missing spacing metadata"
        if not reg_origin or not hr_origin:
            return False, "Missing origin metadata"
            
        # Check spacing
        for r_s, h_s in zip(reg_spacing, hr_spacing):
            if not math.isclose(r_s, h_s, abs_tol=config.spacing_abs_tol, rel_tol=config.spacing_rel_tol):
                return False, f"Spacing mismatch: REG {reg_spacing} vs HR {hr_spacing}"
                
        # Check origin
        for r_o, h_o in zip(reg_origin, hr_origin):
            if not math.isclose(r_o, h_o, abs_tol=config.origin_abs_tol):
                return False, f"Origin mismatch: REG {reg_origin} vs HR {hr_origin}"
                
    if reg_shape != hr_shape:
        if config.shape_mismatch_policy == "allow_only_if_physical_coordinates_match":
            if not config.require_physical_alignment:
                return False, "Shape mismatch and physical alignment not strictly checked."
            # Since physical matches, this is considered OK, but practically requires crops later.
            pass
        else:
            return False, f"Shape mismatch: REG {reg_shape} vs HR {hr_shape}"
            
    return True, "OK"
