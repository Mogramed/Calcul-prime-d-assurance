from __future__ import annotations

from typing import Any, Dict, List

V2_COARSE_CONFIGS: Dict[str, List[Dict[str, Any]]] = {
    "catboost": [
        {
            "config_id": "cb_v2_c1",
            "freq_params": {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 8.0},
            "sev_params": {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 8.0},
            "direct_params": {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 8.0},
        },
        {
            "config_id": "cb_v2_c2",
            "freq_params": {"depth": 7, "learning_rate": 0.025, "l2_leaf_reg": 10.0},
            "sev_params": {"depth": 9, "learning_rate": 0.025, "l2_leaf_reg": 10.0},
            "direct_params": {"depth": 9, "learning_rate": 0.025, "l2_leaf_reg": 10.0},
        },
        {
            "config_id": "cb_v2_c3",
            "freq_params": {"depth": 5, "learning_rate": 0.04, "l2_leaf_reg": 6.0},
            "sev_params": {"depth": 7, "learning_rate": 0.04, "l2_leaf_reg": 6.0},
            "direct_params": {"depth": 7, "learning_rate": 0.04, "l2_leaf_reg": 6.0},
        },
        {
            "config_id": "cb_v2_c4",
            "freq_params": {"depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 12.0},
            "sev_params": {"depth": 8, "learning_rate": 0.02, "l2_leaf_reg": 12.0},
            "direct_params": {"depth": 8, "learning_rate": 0.02, "l2_leaf_reg": 12.0},
        },
        {
            "config_id": "cb_v2_c5",
            "freq_params": {"depth": 8, "learning_rate": 0.02, "l2_leaf_reg": 9.0},
            "sev_params": {"depth": 10, "learning_rate": 0.02, "l2_leaf_reg": 9.0},
            "direct_params": {"depth": 10, "learning_rate": 0.02, "l2_leaf_reg": 9.0},
        },
        {
            "config_id": "cb_v2_c6",
            "freq_params": {"depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 5.0},
            "sev_params": {"depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 5.0},
            "direct_params": {"depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 5.0},
        },
    ],
    "lightgbm": [
        {
            "config_id": "lgb_v2_c1",
            "freq_params": {"num_leaves": 63, "learning_rate": 0.03},
            "sev_params": {"num_leaves": 127, "learning_rate": 0.03},
            "direct_params": {"num_leaves": 127, "learning_rate": 0.03},
        },
        {
            "config_id": "lgb_v2_c2",
            "freq_params": {"num_leaves": 95, "learning_rate": 0.02, "min_child_samples": 80},
            "sev_params": {"num_leaves": 159, "learning_rate": 0.02, "min_child_samples": 60},
            "direct_params": {"num_leaves": 159, "learning_rate": 0.02, "min_child_samples": 60},
        },
        {
            "config_id": "lgb_v2_c3",
            "freq_params": {"num_leaves": 47, "learning_rate": 0.04, "min_child_samples": 40},
            "sev_params": {"num_leaves": 95, "learning_rate": 0.04, "min_child_samples": 40},
            "direct_params": {"num_leaves": 95, "learning_rate": 0.04, "min_child_samples": 40},
        },
        {
            "config_id": "lgb_v2_c4",
            "freq_params": {"num_leaves": 127, "learning_rate": 0.02, "min_child_samples": 120},
            "sev_params": {"num_leaves": 191, "learning_rate": 0.02, "min_child_samples": 80},
            "direct_params": {"num_leaves": 191, "learning_rate": 0.02, "min_child_samples": 80},
        },
        {
            "config_id": "lgb_v2_c5",
            "freq_params": {"num_leaves": 79, "learning_rate": 0.03, "subsample": 0.9},
            "sev_params": {"num_leaves": 127, "learning_rate": 0.03, "subsample": 0.9},
            "direct_params": {"num_leaves": 127, "learning_rate": 0.03, "subsample": 0.9},
        },
        {
            "config_id": "lgb_v2_c6",
            "freq_params": {"num_leaves": 55, "learning_rate": 0.05, "min_child_samples": 60},
            "sev_params": {"num_leaves": 111, "learning_rate": 0.05, "min_child_samples": 50},
            "direct_params": {"num_leaves": 111, "learning_rate": 0.05, "min_child_samples": 50},
        },
    ],
    "xgboost": [
        {
            "config_id": "xgb_v2_c1",
            "freq_params": {"max_depth": 6, "learning_rate": 0.03},
            "sev_params": {"max_depth": 8, "learning_rate": 0.03},
            "direct_params": {"max_depth": 8, "learning_rate": 0.03},
        },
        {
            "config_id": "xgb_v2_c2",
            "freq_params": {"max_depth": 5, "learning_rate": 0.04, "min_child_weight": 8},
            "sev_params": {"max_depth": 7, "learning_rate": 0.04, "min_child_weight": 8},
            "direct_params": {"max_depth": 7, "learning_rate": 0.04, "min_child_weight": 8},
        },
        {
            "config_id": "xgb_v2_c3",
            "freq_params": {"max_depth": 7, "learning_rate": 0.02, "min_child_weight": 4},
            "sev_params": {"max_depth": 9, "learning_rate": 0.02, "min_child_weight": 4},
            "direct_params": {"max_depth": 9, "learning_rate": 0.02, "min_child_weight": 4},
        },
        {
            "config_id": "xgb_v2_c4",
            "freq_params": {"max_depth": 6, "learning_rate": 0.02, "min_child_weight": 12},
            "sev_params": {"max_depth": 8, "learning_rate": 0.02, "min_child_weight": 10},
            "direct_params": {"max_depth": 8, "learning_rate": 0.02, "min_child_weight": 10},
        },
        {
            "config_id": "xgb_v2_c5",
            "freq_params": {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 6},
            "sev_params": {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 6},
            "direct_params": {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 6},
        },
        {
            "config_id": "xgb_v2_c6",
            "freq_params": {"max_depth": 8, "learning_rate": 0.02, "min_child_weight": 3},
            "sev_params": {"max_depth": 10, "learning_rate": 0.02, "min_child_weight": 3},
            "direct_params": {"max_depth": 10, "learning_rate": 0.02, "min_child_weight": 3},
        },
    ],
}


V2_SCREENING_FAMILIES: List[Dict[str, Any]] = [
    {"family": "two_part_classic", "severity_mode": "classic", "tweedie_power": 1.5},
    {"family": "two_part_classic", "severity_mode": "weighted_tail", "tweedie_power": 1.5},
    {"family": "two_part_classic", "severity_mode": "winsorized", "tweedie_power": 1.5},
    {"family": "two_part_tweedie", "severity_mode": "classic", "tweedie_power": 1.3},
    {"family": "two_part_tweedie", "severity_mode": "classic", "tweedie_power": 1.5},
    {"family": "two_part_tweedie", "severity_mode": "classic", "tweedie_power": 1.7},
    {"family": "direct_tweedie", "severity_mode": "classic", "tweedie_power": 1.5},
]
