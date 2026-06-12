"""
Pydantic schemas for all TRACE-CT configurations.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Literal

class DatasetConfig(BaseModel):
    root: str
    dataset_dir: str
    format: str = "ome_zarr"
    volume_manifest: str
    reg_rel_template: str
    hr_rel_template: str
    volume_id_source: str = "manifest"
    
    reg_level: str
    hr_level: str
    axes: List[str]
    patch_size: Tuple[int, int]
    context_offsets: List[int]
    
    use_hr_for_training: bool
    use_hr_for_validation: bool

class SplitsConfig(BaseModel):
    split_file_relative_to_dataset_root: bool
    split_file: str
    split_key: str
    require_donor_receiver_volume_disjoint: bool
    require_train_val_volume_disjoint: bool

class ClipConfig(BaseModel):
    enabled: bool
    min: float
    max: float

class ScalingConfig(BaseModel):
    mode: Literal["zscore_after_clip", "minmax_after_clip"]
    statistics_scope: Literal["per_volume", "global"]
    mean_source: Literal["REG", "HR"]
    std_source: Literal["REG", "HR"]
    apply_same_transform_to_hr: bool

class NormalizationSafetyConfig(BaseModel):
    min_std: float
    on_zero_std: str
    on_reg_hr_inconsistent: str

class NormalizationConfig(BaseModel):
    input_unit: str
    clip: ClipConfig
    scaling: ScalingConfig
    enforce_reg_hr_consistency: bool
    safety: NormalizationSafetyConfig

class AlignmentConfig(BaseModel):
    require_physical_alignment: bool
    shape_mismatch_policy: str
    resampling_policy: str
    crop_policy: str
    spacing_abs_tol: float
    spacing_rel_tol: float
    origin_abs_tol: float
    fallback_behavior: str
    disable_hr_validation_if_alignment_fails: bool

class BoundedAuditConfig(BaseModel):
    max_volumes: int
    max_slices_per_volume: int
    max_patches_per_slice: int
    max_total_patches: int
    require_first_middle_last_slices: bool
    verify_slice_continuity: bool
    verify_chunk_not_used_as_patch: bool
    hash_mode: str
    sample_hash_algorithm: str
    metadata_hash_algorithm: str
    full_content_hash_required: bool

class TraceCTDatasetYAML(BaseModel):
    dataset: DatasetConfig
    splits: SplitsConfig
    normalization: NormalizationConfig
    alignment: AlignmentConfig
    bounded_audit: BoundedAuditConfig

class StageBudget(BaseModel):
    max_steps: Optional[int] = None
    max_cycles: Optional[int] = None
    max_steps_per_cycle: Optional[int] = None
    max_candidate_patches: Optional[int] = None
    max_accepted_patches: Optional[int] = None
    max_donor_volumes: Optional[int] = None

class ProtocolConfig(BaseModel):
    stage_budgets: Dict[str, StageBudget]

class ResidualAuditThresholds(BaseModel):
    min_accepted_count: int
    min_accepted_rate: float
    max_structural_score_q95: float
    max_low_frequency_leakage_q95: float
    max_edge_leakage_q95: float
    relative_std_min: float
    relative_std_max: float
    require_donor_receiver_isolation: bool
    require_residual_pool_freshness: bool
    max_swd: float = 2.0
    max_nps_shape_error: float = 0.15

class ContextAuditThresholds(BaseModel):
    max_high_frequency_correlation: float
    max_high_frequency_leakage_ratio: float
    min_context_helpfulness_delta: float
    max_context_gate_on_unreliable_region: float

class ProposalQualificationThresholds(BaseModel):
    min_homogeneous_noise_reduction: float
    min_edge_contrast_retention: float
    min_lesion_contrast_retention: float
    max_low_frequency_bias: float
    max_nps_shape_error: float
    reject_gaussian_blur_negative_control: bool

class DynamicTargetGatingThresholds(BaseModel):
    max_W_fb_in_lesion_mean: float
    max_W_fb_at_edge_mean: float
    min_W_fb_in_homogeneous_mean: float
    max_target_drift_q95: float
    max_pd_disagreement_allowed: float
    reject_all_ones_gate: bool
    all_ones_gate_threshold: float = 0.95
    rho_ramp_step: float = 0.05
    max_rho: float = 1.0
    initial_alpha: float = 0.5

class CycleStabilityThresholds(BaseModel):
    max_cycle_drift_q95: float
    max_target_drift_q95: float
    max_residual_policy_change: float
    max_loss_oscillation_ratio: float

class ThresholdsYAML(BaseModel):
    residual_audit: ResidualAuditThresholds
    context_audit: ContextAuditThresholds
    proposal_qualification: ProposalQualificationThresholds
    dynamic_target_gating: DynamicTargetGatingThresholds
    cycle_stability: CycleStabilityThresholds
