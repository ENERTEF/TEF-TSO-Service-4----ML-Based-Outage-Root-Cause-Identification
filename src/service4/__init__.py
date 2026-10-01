"""Reusable Service 4 incident-grouping functionality."""

from service4.evaluation import (
    add_minute_precision_reference_bounds,
    add_temporal_overlap_diagnostics,
    bcubed_scores,
    build_exact_asset_candidates,
    clustering_metric_catalog,
    evaluate_episode_clusters,
    evaluate_event_partition,
    evaluate_event_partition_by_group,
    match_clusters_by_membership,
    summarize_candidate_matches_by_group,
    summarize_temporal_matches,
    summarize_temporal_matches_by_group,
)
from service4.state_machine_v1 import (
    ASSET_EPISODE_GRAMMAR_VERSION,
    MAX_EPISODE_DURATION,
    RETRY_GRACE,
    StateMachineV1Result,
    attach_v1_control_context,
    build_state_machine_v1_episodes,
    prepare_state_machine_v1_events,
    run_state_machine_v1,
    select_v1_control_context,
)
from service4.state_machine_v1_1 import (
    ASSET_EPISODE_GRAMMAR_VERSION as ASSET_EPISODE_GRAMMAR_VERSION_V1_1,
)
from service4.state_machine_v1_1 import (
    CONTROL_RESTORATION_RULE,
    StateMachineV1_1Result,
    build_state_machine_v1_1_episodes,
    run_state_machine_v1_1,
    select_control_confirmed_restorations,
)
from service4.state_machine_v2 import (
    ASSET_EPISODE_GRAMMAR_VERSION as ASSET_EPISODE_GRAMMAR_VERSION_V2,
)
from service4.state_machine_v2 import (
    StateMachineV2Result,
    build_state_machine_v2_episodes,
    run_state_machine_v2,
)

__all__ = [
    "add_minute_precision_reference_bounds",
    "add_temporal_overlap_diagnostics",
    "bcubed_scores",
    "build_exact_asset_candidates",
    "clustering_metric_catalog",
    "evaluate_episode_clusters",
    "evaluate_event_partition",
    "evaluate_event_partition_by_group",
    "match_clusters_by_membership",
    "summarize_candidate_matches_by_group",
    "summarize_temporal_matches",
    "summarize_temporal_matches_by_group",
    "ASSET_EPISODE_GRAMMAR_VERSION",
    "MAX_EPISODE_DURATION",
    "RETRY_GRACE",
    "StateMachineV1Result",
    "attach_v1_control_context",
    "build_state_machine_v1_episodes",
    "prepare_state_machine_v1_events",
    "run_state_machine_v1",
    "select_v1_control_context",
    "ASSET_EPISODE_GRAMMAR_VERSION_V1_1",
    "CONTROL_RESTORATION_RULE",
    "StateMachineV1_1Result",
    "build_state_machine_v1_1_episodes",
    "run_state_machine_v1_1",
    "select_control_confirmed_restorations",
    "ASSET_EPISODE_GRAMMAR_VERSION_V2",
    "StateMachineV2Result",
    "build_state_machine_v2_episodes",
    "run_state_machine_v2",
]
