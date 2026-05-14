# CKA-Guided Dynamic Sparse Federated Learning

## Research Objective

This repository is a research prototype for simulated federated learning on MNIST under non-IID client data. The goal is to compare dense, fixed-sparse, dynamic-sparse, and representation-guided sparse training methods in a controlled PyTorch codebase.

The proposed direction is to use layer-wise client representation similarity, measured with linear Centered Kernel Alignment (CKA), to guide how sparsity is allocated across layers during dynamic sparse federated learning.

## Methods Compared

This project currently supports four methods:

1. **FedAvg**
   Dense federated averaging. Each client trains a local copy of the global model, and the server aggregates client weights weighted by client dataset size.

2. **Sparse FedAvg**
   FedAvg with fixed unstructured binary masks. Masks are applied to trainable weight tensors only, not biases. The same sparse topology is preserved throughout training.

3. **FedDST**
   A simple dynamic sparse training baseline inspired by RigL. At scheduled rounds, each layer prunes low-magnitude active weights and regrows inactive weights with high accumulated gradient magnitude.

4. **CKA-guided FedDST**
   The proposed prototype method. It periodically computes client representation similarity with CKA and converts layer-wise CKA scores into adaptive layer-wise sparsity targets used by the dynamic sparse update.

## Dataset and Non-IID Setup

Experiments use MNIST through `torchvision`.

Training data is split across simulated clients using Dirichlet label-skew partitioning:

- `num_clients`: number of simulated clients
- `alpha`: Dirichlet concentration parameter
- lower `alpha` gives stronger non-IID label skew
- `seed` controls deterministic partitioning

The test set remains global. A small balanced reference set is also sampled from the test data for CKA computation.

## Model Architecture

The default model is `SmallCNN` for MNIST-like grayscale images:

- `conv1`: 1 input channel to 32 channels
- `conv2`: 32 channels to 64 channels
- `fc1`: hidden fully connected layer
- `fc2`: classifier output layer

The forward pass can optionally return activations from:

- `conv1`
- `conv2`
- `fc1`

These activations are used for CKA analysis.

## Sparsity Method

Sparse methods use unstructured binary masks over trainable weight tensors:

- biases are not sparsified
- masks are applied after initialization
- masks are reapplied after every optimizer step
- masks are reapplied after server aggregation

The code logs:

- total sparsity
- layer-wise sparsity
- active parameter count
- pruned and regrown weights for dynamic methods

## CKA-Guided Layer-Wise Sparsity

CKA-guided FedDST computes pairwise client CKA on the shared reference loader every `cka_interval` rounds.

For each layer:

1. Client models pass the reference data through the network.
2. Activations from `conv1`, `conv2`, and `fc1` are flattened.
3. Linear CKA is computed pairwise across client models.
4. The average upper-triangle CKA score is used as the layer similarity score.

Layer-wise sparsity targets are then adapted:

- high CKA layers receive lower sparsity
- low CKA layers receive higher sparsity
- the total active parameter budget remains approximately fixed at the base sparsity
- classifier-only layers such as `fc2` are not assigned CKA-guided targets

These targets are passed into the FedDST pruning and regrowth step.

## How to Run Experiments

Experiments are configured with YAML files. Shared settings live in
`configs/global.yaml`; method-specific files only contain values that differ
from the shared setup.

Run a single method:

```bash
python -m src.train --config configs/fedavg_mnist.yaml
```

Run sparse FedAvg:

```bash
python -m src.train --config configs/sparse_fedavg_mnist.yaml
```

Run FedDST:

```bash
python -m src.train --config configs/feddst_mnist.yaml
```

Run CKA-guided FedDST:

```bash
python -m src.train --config configs/cka_feddst_mnist.yaml
```

Command-line arguments override YAML values only when explicitly provided:

```bash
python -m src.train --config configs/cka_feddst_mnist.yaml --rounds 5 --sparsity 0.9
```

Run all four MNIST methods sequentially:

```bash
python experiments/run_all_mnist.py
```

Run the main sparsity sweep. This runs dense FedAvg once, then Sparse FedAvg,
FedDST, and CKA-FedDST at `0.5, 0.7, 0.8, 0.9, 0.95` sparsity:

```bash
python experiments/run_experiment.py
```

Preview the sparsity sweep commands without launching experiments:

```bash
python experiments/run_experiment.py --dry_run
```

Run selected sparsity levels or methods:

```bash
python experiments/run_experiment.py --sparsities 0.8 0.9
python experiments/run_experiment.py --methods feddst cka_feddst
```

Generate plots from saved logs:

```bash
python experiments/plot_results.py
python experiments/plot_results.py --log_dir results/logs --plot_dir results/plots
```

## Logs and Plots

Training logs are saved to:

```text
results/logs/
```

Each run also saves the final merged configuration next to the CSV log for
reproducibility.

Logs include, depending on the method:

- round
- test accuracy
- test loss
- average training loss
- total sparsity
- layer-wise sparsity
- pruned and regrown weights
- mask changes
- layer-wise CKA
- CKA-guided target sparsity

Plots are saved to:

```text
results/plots/
```

Generated plots include:

- accuracy vs communication rounds for all method-sparsity combinations
- final accuracy vs sparsity
- best accuracy vs sparsity
- accuracy vs communication cost when a cost proxy is logged
- CKA-FedDST layer-wise sparsity vs rounds
- CKA-FedDST layer-wise CKA vs rounds

## Current Limitations

This is a research prototype, not a final benchmark suite.

Current limitations:

- experiments are simulated on one machine
- only MNIST is fully wired into the data pipeline
- the model is a small CNN
- client selection is currently full participation
- hyperparameters are simple defaults, not tuned
- CKA-guided target allocation is heuristic
- no explicit communication-cost accounting yet
- no checkpoint-based experiment resume support yet

## Future Work

Planned extensions:

- add Fashion-MNIST experiments
- add CIFAR-10 experiments
- add ResNet-style models
- scale to more clients
- run more random seeds
- add communication cost analysis
- compare different CKA intervals and target-allocation strengths
- evaluate partial client participation
- add checkpointing and experiment resume support

## Repository Structure

```text
configs/                 Experiment configuration files
experiments/             Experiment and plotting scripts
results/
  logs/                  CSV training and CKA logs
  checkpoints/           Saved model checkpoints
  plots/                 Generated figures
src/
  data.py                MNIST loading and non-IID partitioning
  models.py              SmallCNN model definition
  federated.py           FedAvg, Sparse FedAvg, FedDST, CKA-FedDST loops
  sparsity.py            Mask creation and sparsity utilities
  dst.py                 Dynamic sparse pruning/regrowth
  cka.py                 Linear CKA computation
  config.py              YAML loading and configuration merging
  train.py               Training CLI
  evaluate.py            Evaluation utilities
  plotting.py            Plotting utilities
  utils.py               Shared helpers
```
