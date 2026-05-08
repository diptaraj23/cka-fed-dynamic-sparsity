# Simulated Federated Learning on MNIST

This repository is a lightweight research scaffold for simulated federated learning experiments on MNIST. The goal is to provide a clean place to add data loading, model definitions, federated orchestration, sparsity methods, dynamic sparse training, CKA analysis, evaluation, and plotting.

The full method is intentionally not implemented yet. Current files define small, documented entry points and placeholders so experiments can grow in a modular way.

## Structure

```text
configs/                 Experiment configuration files
experiments/             Scripts or notebooks for individual studies
results/
  logs/                  Training and evaluation logs
  checkpoints/           Saved model checkpoints
  plots/                 Figures and analysis outputs
src/
  data.py                Dataset and client partition helpers
  models.py              PyTorch model builders
  federated.py           Federated simulation utilities
  sparsity.py            Sparsity helpers
  dst.py                 Dynamic sparse training hooks
  cka.py                 CKA analysis helpers
  train.py               Training command entry point
  evaluate.py            Evaluation utilities
  plotting.py            Plotting helpers
  utils.py               Shared utilities
```

## Quick Check

```bash
python -m src.train --help
```

This should print the available command-line options without starting an experiment.

## Research Notes

Keep new code small, explicit, and easy to inspect. Prefer simple functions and dataclasses until repeated experiment patterns justify additional abstraction.
