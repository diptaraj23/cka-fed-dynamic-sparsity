"""Lightweight smoke tests for the research prototype.

These tests avoid dataset downloads and long training. They are intended to
catch import, configuration, model, sparsity, and experiment-runner regressions.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.run_experiment import build_parser as build_runner_parser
from experiments.run_experiment import build_run_specs
from src.cka import linear_cka
from src.data import (
    DataConfig,
    get_subset_indices,
    make_balanced_reference_dataset,
    partition_clients,
)
from src.models import get_model
from src.sparsity import (
    SparsityConfig,
    cka_scores_to_layer_sparsities,
    create_masks,
    sparsity_summary,
)
from src.train import build_parser as build_train_parser
from src.train import load_final_config


class ToyDataset:
    """Small labeled dataset with the target interface used by torchvision."""

    def __init__(self) -> None:
        self.targets = np.repeat(np.arange(10), 20)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index):
        return torch.zeros(1, 28, 28), int(self.targets[index])


class ResearchPipelineSmokeTests(unittest.TestCase):
    """Fast checks for the main research workflow pieces."""

    def test_model_forward_returns_expected_activations(self) -> None:
        model = get_model("small_cnn", "mnist")
        inputs = torch.randn(4, 1, 28, 28)

        logits, activations = model(inputs, return_activations=True)

        self.assertEqual(tuple(logits.shape), (4, 10))
        self.assertEqual({"conv1", "conv2", "fc1"}, set(activations))
        self.assertEqual(activations["conv1"].shape[0], 4)
        self.assertEqual(activations["conv2"].shape[0], 4)
        self.assertEqual(tuple(activations["fc1"].shape), (4, 128))

    def test_sparsity_masks_exclude_biases_and_hold_target(self) -> None:
        model = get_model("small_cnn", "mnist")
        masks = create_masks(
            model,
            SparsityConfig(target_sparsity=0.8, init_method="random", seed=123),
        )
        summary = sparsity_summary(model, masks)

        self.assertTrue(masks)
        self.assertTrue(all(name.endswith(".weight") for name in masks))
        self.assertFalse(any("bias" in name for name in masks))
        self.assertAlmostEqual(summary["total_sparsity"], 0.8, places=3)

    def test_cka_guidance_does_not_target_classifier_without_cka(self) -> None:
        model = get_model("small_cnn", "mnist")
        masks = create_masks(model, SparsityConfig(target_sparsity=0.8, seed=123))

        targets = cka_scores_to_layer_sparsities(
            masks=masks,
            cka_scores={"conv1": 0.9, "conv2": 0.6, "fc1": 0.3},
            base_sparsity=0.8,
            strength=1.0,
            min_sparsity=0.5,
            max_sparsity=0.95,
        )

        self.assertIn("conv1.weight", targets)
        self.assertIn("conv2.weight", targets)
        self.assertIn("fc1.weight", targets)
        self.assertNotIn("fc2.weight", targets)

    def test_linear_cka_is_finite_and_bounded(self) -> None:
        torch.manual_seed(0)
        score = linear_cka(torch.randn(12, 8), torch.randn(12, 8))

        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_reference_indices_are_excluded_from_client_partitions(self) -> None:
        dataset = ToyDataset()
        config = DataConfig(num_clients=5, alpha=0.5, seed=7, reference_size=50)
        reference = make_balanced_reference_dataset(dataset, size=50, seed=7)
        reference_indices = set(get_subset_indices(reference))

        clients = partition_clients(dataset, config, exclude_indices=reference_indices)
        client_indices = {
            index
            for client in clients
            for index in get_subset_indices(client)
        }

        self.assertTrue(all(len(client) > 0 for client in clients))
        self.assertTrue(reference_indices.isdisjoint(client_indices))

    def test_yaml_cli_overrides_take_precedence(self) -> None:
        parser = build_train_parser()
        args = parser.parse_args(
            [
                "--config",
                "configs/cka_feddst_mnist.yaml",
                "--rounds",
                "5",
                "--sparsity",
                "0.9",
            ]
        )

        config = load_final_config(args)

        self.assertEqual(config["rounds"], 5)
        self.assertEqual(config["sparsity"], 0.9)
        self.assertEqual(config["num_clients"], 5)
        self.assertEqual(config["method"], "cka_feddst")

    def test_default_suite_run_counts_are_explicit(self) -> None:
        parser = build_runner_parser()

        multiseed_args = parser.parse_args(["--suite", "multiseed"])
        cka_args = parser.parse_args(["--suite", "cka_strength"])
        all_args = parser.parse_args(["--suite", "all"])

        self.assertEqual(len(build_run_specs(multiseed_args, "test_suite")), 80)
        self.assertEqual(len(build_run_specs(cka_args, "test_suite")), 125)
        self.assertEqual(len(build_run_specs(all_args, "test_suite")), 205)


if __name__ == "__main__":
    unittest.main()
