# Results Folder Guide

`results/` stores experiment artifacts. Source code, configs, and tests live outside this folder.

## Layout

- `logs/`: raw per-run training outputs. Each suite folder contains a `manifest.csv`, run CSVs, CKA CSVs, merged config YAMLs, and split manifests.
- `averaged/`: compact mean/std summaries generated from `logs/` by `experiments/aggregate_results.py`.
- `plots/`: figures generated from `averaged/` by `experiments/plot_results.py`.
- `checkpoints/`: optional model checkpoints when checkpoint saving is enabled. This repo normally keeps only `.gitkeep` here.
- `_run_all_tmp/`: temporary audit output. Not part of the main experiment.

## Suite Folders

- `multiseed/<dataset>` or `multiseed/<dataset>_<suite>_<timestamp>`: compares FedAvg, Sparse FedAvg, FedDST, and CKA-FedDST across seeds and sparsities.
- `cka_strength_sweep/<dataset>` or `cka_strength_sweep/<dataset>_<suite>_<timestamp>`: sweeps CKA guidance strength for CKA-FedDST.

Current curated CIFAR-10 outputs use the compact suite id `cifar10`:

- `logs/multiseed/cifar10`
- `logs/cka_strength_sweep/cifar10`

Original MNIST outputs are named with the `mnist_` prefix so they are distinguishable from Fashion-MNIST and CIFAR-10.

## Key Files

- `manifest.csv`: planned commands, output paths, and pass/fail status.
- `*.csv`: round-level training metrics.
- `*_cka.csv`: layer-wise CKA measurements.
- `*_config.yaml`: exact merged configuration for reproducibility.
- `splits/*.json`: client/reference split manifest.

## Regenerating Summaries

```bash
python experiments/aggregate_results.py
python experiments/plot_results.py
```

For one suite:

```bash
python experiments/aggregate_results.py --suite multiseed --log-dir results/logs/multiseed/cifar10
python experiments/plot_results.py --avg-dir results/averaged/multiseed/cifar10 --plot-dir results/plots/multiseed/cifar10
```
