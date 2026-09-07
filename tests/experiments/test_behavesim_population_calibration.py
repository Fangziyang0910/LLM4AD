import numpy as np

from experiments.behavesim_population_calibration.run import (
    compare_matrices,
    landscape,
)
from experiments.traceaad_refine_e1 import profile_core as core


def test_obp_matched_probe_uses_training_lengths_and_capacities():
    compact = core._build_probe_data("online_bin_packing", "A", obp_scale="compact")
    matched = core._build_probe_data("online_bin_packing", "A", obp_scale="matched")

    assert sorted(len(value["items"]) for value in compact["instances"]) == [256] * 4
    assert sorted(len(value["items"]) for value in matched["instances"]) == [1000, 1000, 5000, 5000]
    assert sorted(value["capacity"] for value in matched["instances"]) == [100, 100, 500, 500]
    assert matched["probe_metadata"]["scale"] == "matched"


def test_aco_probe_records_all_fixed_random_streams():
    offsets = core.CALIBRATED_ACO_SEED_OFFSETS
    data = core._build_probe_data("op_aco", "A", aco_seed_offsets=offsets)

    assert data["probe_metadata"]["aco_seed_offsets"] == list(offsets)
    assert data["probe_metadata"]["instances"] == 4
    assert data["probe_metadata"]["role"] == "training-probe"


def test_landscape_statistics_detect_ordered_population_geometry():
    positions = np.arange(8, dtype=float)
    matrix = np.abs(positions[:, None] - positions[None, :])
    result = landscape(matrix, positions)

    assert result["distance_fitness_gap_spearman"] > 0.98
    assert result["random_minus_knn_gap"] > 0
    assert result["far_minus_near_gap"] > 0
    comparison = compare_matrices(matrix, matrix, k=2)
    assert np.isclose(comparison["distance_spearman"], 1.0)
    assert comparison["knn_overlap"] == 1.0
