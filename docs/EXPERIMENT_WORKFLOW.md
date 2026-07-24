# Experiment Workflow

This document explains how an experiment moves from configuration files to raw
logs, averaged CSV files, plots, and final conclusions. It is meant to make the
MNIST and Fashion-MNIST results traceable and easier to explain.

## 1. Configuration

Experiments start from YAML configuration files.

- `configs/global.yaml` contains shared MNIST settings.
- `configs/global_fashion_mnist.yaml` contains shared Fashion-MNIST settings.
- Method-specific configs, such as `configs/feddst_mnist.yaml` and
  `configs/cka_feddst_mnist.yaml`, contain only method-specific options.

`src/train.py` merges configuration in this order:

1. Built-in defaults from `src/config.py`
2. Global YAML config
3. Method-specific YAML config
4. Explicit command-line overrides

The final merged configuration is printed at the start of training and saved
next to the run log as `<run_name>_config.yaml`.

## 2. Experiment Suites

The main runner is:

```bash
python experiments/run_experiment.py
```

Important suite types:

- `sparsity`: one seed, dense FedAvg once, and sparse methods over sparsities.
- `multiseed`: all four methods across all selected seeds and sparsities.
- `cka_strength`: only CKA-FedDST across selected seeds, sparsities, and CKA
  strengths.
- `all`: runs `multiseed` first, then `cka_strength`.

For Fashion-MNIST:

```bash
python experiments/run_experiment.py --dataset fashion_mnist --suite all
```

The runner writes isolated suite folders, for example:

```text
results/logs/multiseed/<suite_id>/seed_42/
results/logs/cka_strength_sweep/<suite_id>/strength_0p8/seed_42/
```

Each suite also writes a `manifest.csv` that records the command, method, seed,
sparsity, CKA strength, output folders, and run status.

## 3. Data Pipeline

Data loading is handled by `src/data.py`.

The workflow is:

1. Load MNIST or Fashion-MNIST from `torchvision`.
2. Reserve a small balanced reference subset from the training data for CKA.
3. Exclude the reference examples from client training.
4. Partition the remaining training data across clients using Dirichlet
   label-skew sampling.
5. Build one training `DataLoader` per client.
6. Build one global test `DataLoader`.
7. Build one shared CKA reference `DataLoader`.

The key non-IID parameter is `alpha`.

- Lower `alpha` means stronger label skew.
- Higher `alpha` means more IID-like client distributions.

The current experiments use:

```text
num_clients = 5
alpha = 0.3
reference_size = 200
```

Every run saves a split manifest JSON file containing exact client indices,
reference indices, and label distributions. This makes the partition traceable.

## 4. Training Workflow

The selected method is executed by `src/train.py`, which calls the appropriate
function in `src/federated.py`.

### FedAvg

For each communication round:

1. Copy the global model to each client.
2. Train each client locally.
3. Send client weights back to the server.
4. Aggregate weights with FedAvg, weighted by client dataset size.
5. Evaluate the global model on the test set.
6. Log round metrics.

### Sparse FedAvg

Sparse FedAvg follows the same FedAvg workflow but applies binary masks to
trainable weight tensors.

Masks are applied:

1. After sparse initialization
2. After every local optimizer step
3. After server aggregation

Bias parameters are not sparsified.

### FedDST

FedDST adds dynamic sparse mask updates.

At the configured `mask_update_interval`, the method:

1. Prunes low-magnitude active weights.
2. Regrows inactive weights with large gradient magnitude.
3. Keeps the active parameter count approximately fixed.
4. Logs pruned weights, regrown weights, mask changes, total sparsity, and
   layer-wise sparsity.

### CKA-FedDST

CKA-FedDST follows FedDST but periodically computes CKA and converts CKA scores
into layer-wise sparsity targets.

The CKA-guided update is:

```text
client models
-> shared reference data
-> layer activations
-> pairwise CKA matrices
-> average layer CKA
-> layer-wise sparsity targets
-> DST pruning/regrowth
```

The current CKA layers are:

```text
conv1
conv2
fc1
```

The classifier layer `fc2.weight` is not directly assigned a CKA-guided target.

## 5. Pairwise CKA Computation

CKA computation is implemented in `src/cka.py`.

For each client model:

1. Pass the shared reference loader through the model.
2. Collect activations from `conv1`, `conv2`, and `fc1`.
3. Flatten each activation tensor to shape `[num_samples, features]`.
4. Compute linear CKA between every pair of client activation matrices.

For 5 clients, each layer produces a `5 x 5` matrix:

```text
          c0    c1    c2    c3    c4
c0      1.00  ...   ...   ...   ...
c1      ...   1.00  ...   ...   ...
c2      ...   ...   1.00  ...   ...
c3      ...   ...   ...   1.00  ...
c4      ...   ...   ...   ...   1.00
```

The diagonal is always a client compared with itself, so it is not used for the
layer summary.

## 6. CKA Averaging Within One Run

For each layer, the code averages the strict upper triangle of the pairwise CKA
matrix.

For 5 clients, the averaged pairs are:

```text
c0 vs c1
c0 vs c2
c0 vs c3
c0 vs c4
c1 vs c2
c1 vs c3
c1 vs c4
c2 vs c3
c2 vs c4
c3 vs c4
```

That gives 10 pairwise values per layer.

The per-run layer CKA is:

```text
average_layer_cka = mean(CKA(client_i, client_j)) for all i < j
```

These detailed pairwise CKA rows are saved in files ending with:

```text
_cka.csv
```

The pairwise CKA schema is documented in:

```text
docs/log_schemas/cka_pairwise_log_schema.csv
```

## 7. CKA-Guided Sparsity Targets

CKA-FedDST converts layer CKA scores into sparsity targets in `src/sparsity.py`.

The rule is:

- Higher CKA means clients agree more on that layer.
- Higher-CKA layers receive lower sparsity, preserving more weights.
- Lower-CKA layers receive higher sparsity.
- The active parameter budget remains approximately equal to the base sparsity.

Only layers with measured CKA scores receive adaptive targets. In the current
model, that means:

```text
conv1.weight
conv2.weight
fc1.weight
```

The final classifier layer:

```text
fc2.weight
```

is kept out of direct CKA-guided adaptation because there is no CKA activation
score for it.

## 8. Raw Training Logs

Each training run saves a CSV log under the selected `log_dir`.

Common columns include:

- `method`
- `dataset`
- `sparsity`
- `seed`
- `cka_strength`
- `round`
- `test_accuracy`
- `test_loss`
- `avg_train_loss`
- `split_manifest_path`

Sparse methods also log:

- `total_sparsity`
- `active_params`
- `total_params`
- `layer_sparsity`
- `sparsity_conv1_weight`
- `sparsity_conv2_weight`
- `sparsity_fc1_weight`
- `sparsity_fc2_weight`

Dynamic sparse methods also log:

- `pruned_weights`
- `regrown_weights`
- `mask_changes`

CKA-FedDST additionally logs:

- `cka_computed`
- `layer_cka`
- `layer_target_sparsity`
- `cka_conv1`
- `cka_conv2`
- `cka_fc1`
- `target_sparsity_conv1_weight`
- `target_sparsity_conv2_weight`
- `target_sparsity_fc1_weight`

Column meanings are documented in `docs/log_schemas/`.

## 9. Aggregation Across Seeds

Raw logs are aggregated by:

```bash
python experiments/aggregate_results.py
```

The aggregation code lives in `src/aggregation.py`.

The script reads raw logs from suite folders and writes averaged CSV files to:

```text
results/averaged/<suite>/<suite_id>/
```

The main averaged outputs are:

```text
round_metrics_mean_std.csv
final_metrics_mean_std.csv
best_metrics_mean_std.csv
layerwise_sparsity_mean_std.csv
layerwise_cka_mean_std.csv
aggregation_manifest.csv
```

### Round Metrics

`round_metrics_mean_std.csv` groups by:

```text
method, dataset, sparsity, cka_strength, round
```

For each numeric metric, it computes:

```text
mean
std
count
seed_count
seeds
```

This is how curves such as accuracy vs communication rounds are produced.

### Final Metrics

`final_metrics_mean_std.csv` first selects the final evaluated row for each run.

It then groups by:

```text
method, dataset, sparsity, cka_strength
```

This is used for final accuracy vs sparsity plots.

### Best Metrics

`best_metrics_mean_std.csv` selects the best test-accuracy row from each run.

It groups by:

```text
method, dataset, sparsity, cka_strength
```

This is used for best accuracy vs sparsity plots.

### Layer-Wise Sparsity and CKA

Layer-wise files group by:

```text
method, dataset, sparsity, cka_strength, round, layer
```

They report mean/std across seeds for:

- actual layer sparsity
- CKA-guided target sparsity
- layer-wise CKA

## 10. Plot Generation

Plots are generated by:

```bash
python experiments/plot_results.py
```

The plotting script reads averaged CSV files, not raw logs, for final research
figures.

Default output location:

```text
results/plots/<suite>/<suite_id>/
```

Important plots include:

- `accuracy_vs_rounds_mean_std.png`
- `final_accuracy_mean_std_vs_sparsity.png`
- `best_accuracy_mean_std_vs_sparsity.png`
- `cka_strength_final_accuracy_vs_sparsity.png`
- `cka_strength_best_accuracy_vs_sparsity.png`
- `cka_feddst_layerwise_sparsity.png`
- `cka_feddst_layerwise_cka.png`

## 11. How One Final Plot Point Is Computed

Example point:

```text
Fashion-MNIST, FedDST, sparsity = 0.95, final accuracy
```

The computation path is:

1. Run FedDST on Fashion-MNIST at sparsity `0.95` for each seed.
2. Each run logs `test_accuracy` at every evaluated communication round.
3. The aggregation script selects the final evaluated row from each seed.
4. The five final accuracy values are grouped by:

```text
method = feddst
dataset = fashion_mnist
sparsity = 0.95
cka_strength = blank
```

5. The aggregation script computes:

```text
mean final accuracy
standard deviation
seed count
seed list
```

6. `plot_results.py` reads `final_metrics_mean_std.csv`.
7. The point is drawn on `final_accuracy_mean_std_vs_sparsity.png`.

## 12. How One CKA Plot Point Is Computed

Example point:

```text
CKA-FedDST, Fashion-MNIST, sparsity = 0.8, layer = fc1, round = 20
```

The computation path is:

1. At a CKA round, each client model processes the shared reference dataset.
2. `fc1` activations are flattened for every client.
3. Linear CKA is computed for every client pair.
4. The strict upper-triangle client-pair values are averaged within the run.
5. The resulting `average_layer_cka` is logged.
6. Across seeds, aggregation groups by:

```text
method, dataset, sparsity, cka_strength, round, layer
```

7. The mean and standard deviation across seeds are written to:

```text
layerwise_cka_mean_std.csv
```

8. `plot_results.py` draws the averaged layer-wise CKA curve.

## 13. Result Interpretation Workflow

The final conclusions are based on these aggregated outputs:

1. Compare dense FedAvg with sparse methods to understand the cost of sparsity.
2. Compare Sparse FedAvg with FedDST to evaluate dynamic mask updates.
3. Compare FedDST with CKA-FedDST to evaluate the CKA-guided adaptation.
4. Inspect total sparsity and active parameter counts to confirm the sparsity
   budget is controlled.
5. Inspect layer-wise CKA to determine whether CKA provides useful contrast
   between layers.
6. Inspect CKA-strength sweeps to determine whether stronger CKA guidance
   changes performance.

In the current results, dynamic sparse training is strongly useful at high
sparsity, but CKA guidance does not consistently improve over FedDST because
the measured layer-wise CKA values are highly saturated.

