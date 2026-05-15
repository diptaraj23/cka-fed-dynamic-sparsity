# Plot Explanation and Result Analysis

This folder contains plots generated from the CSV logs in `results/logs/`.
The plots summarize one MNIST non-IID federated learning sweep with:

- dataset: MNIST
- clients: 5
- Dirichlet alpha: 0.3
- seed: 42
- rounds: 20
- local epochs: 1
- sparse methods tested at sparsity: 0.5, 0.7, 0.8, 0.9, 0.95

The dense FedAvg run is included as a reference baseline with sparsity 0.0.

## How the Plots Were Generated

The plotting code is in `src/plotting.py` and is launched by:

```bash
python experiments/plot_results.py
```

The plotting script:

1. Reads all main training CSV files from `results/logs/`.
2. Skips pairwise CKA matrix files ending in `_cka.csv` when building the main accuracy and sparsity plots.
3. Infers method and sparsity from CSV columns when available.
4. Falls back to parsing filenames when older logs do not contain metadata columns.
5. Keeps the newest log for each method-sparsity pair if duplicates exist.
6. Saves figures into `results/plots/`.

The current plots were generated on 2026-05-15 from logs created on 2026-05-11.

## Plot: `accuracy_vs_rounds_all_sparsities.png`

### How It Was Calculated

- Source columns: `round`, `test_accuracy`, `method`, `sparsity`
- X-axis: communication round
- Y-axis: test accuracy on the global MNIST test set
- Each curve: one method-sparsity combination
- FedAvg is shown as the dense baseline
- Sparse methods are shown as method plus sparsity, for example `FedDST s=0.9`

### What Each Section Means

- The title states that the figure tracks test accuracy over communication rounds.
- The x-axis shows the number of federated communication rounds completed.
- The y-axis shows global test accuracy after aggregation and evaluation.
- The legend identifies each method and sparsity level.
- The grid helps compare convergence speed and final accuracy across curves.

### Analysis

FedAvg is the dense upper reference and reaches a final accuracy of 0.9124, with a best accuracy of 0.9190 at round 17.

At moderate sparsity, all sparse methods learn reasonably well. At high sparsity, the dynamic sparse methods are much stronger than fixed Sparse FedAvg:

- Sparse FedAvg at 0.9 final accuracy: 0.6244
- FedDST at 0.9 final accuracy: 0.8372
- CKA-FedDST at 0.9 final accuracy: 0.8355
- Sparse FedAvg at 0.95 final accuracy: 0.1135
- FedDST at 0.95 final accuracy: 0.8003
- CKA-FedDST at 0.95 final accuracy: 0.8031

The main visual takeaway should be that fixed sparse training becomes fragile at very high sparsity, while dynamic sparse methods remain usable.

## Plot: `final_accuracy_vs_sparsity.png`

### How It Was Calculated

- Source columns: `round`, `test_accuracy`, `method`, `sparsity`
- Only sparse methods are included:
  - Sparse FedAvg
  - FedDST
  - CKA-FedDST
- For each method-sparsity pair, the final row after sorting by `round` is selected.
- X-axis: target sparsity level
- Y-axis: final test accuracy at the last communication round

### What Each Section Means

- The title says the plot compares final performance under different sparsity levels.
- The x-axis increases from less sparse to more sparse.
- The y-axis shows final test accuracy after 20 rounds.
- Each line is one sparse method.

### Analysis

Final accuracy values:

| Method | s=0.5 | s=0.7 | s=0.8 | s=0.9 | s=0.95 |
|---|---:|---:|---:|---:|---:|
| Sparse FedAvg | 0.8829 | 0.8572 | 0.8205 | 0.6244 | 0.1135 |
| FedDST | 0.8829 | 0.8689 | 0.8456 | 0.8372 | 0.8003 |
| CKA-FedDST | 0.8839 | 0.8708 | 0.8478 | 0.8355 | 0.8031 |

The plot shows a clear sparsity sensitivity pattern:

- At 0.5 sparsity, all sparse methods are similar.
- At 0.7 and 0.8, dynamic sparse methods pull ahead of Sparse FedAvg.
- At 0.9 and 0.95, Sparse FedAvg degrades sharply.
- FedDST and CKA-FedDST remain much more stable under extreme sparsity.

CKA-FedDST is slightly better than FedDST at 0.5, 0.7, 0.8, and 0.95, but slightly lower at 0.9. The difference is small in this single-seed run.

## Plot: `best_accuracy_vs_sparsity.png`

### How It Was Calculated

- Source columns: `round`, `test_accuracy`, `method`, `sparsity`
- Only sparse methods are included.
- For each method-sparsity pair, the row with maximum `test_accuracy` is selected.
- X-axis: target sparsity level
- Y-axis: best test accuracy achieved during training

### What Each Section Means

- The title says the plot compares peak performance, not final performance.
- The x-axis is sparsity.
- The y-axis is the best global test accuracy observed at any round.
- Each line represents one sparse method.

### Analysis

Best accuracy values:

| Method | s=0.5 | s=0.7 | s=0.8 | s=0.9 | s=0.95 |
|---|---:|---:|---:|---:|---:|
| Sparse FedAvg | 0.8887 | 0.8595 | 0.8205 | 0.6244 | 0.1135 |
| FedDST | 0.8872 | 0.8729 | 0.8491 | 0.8372 | 0.8004 |
| CKA-FedDST | 0.8890 | 0.8780 | 0.8525 | 0.8395 | 0.8031 |

This plot is useful because final-round accuracy can be affected by late-round noise. The best-accuracy plot shows CKA-FedDST has the highest peak accuracy at every tested sparsity level in this run.

The improvements over FedDST are modest:

- s=0.5: +0.0018
- s=0.7: +0.0051
- s=0.8: +0.0034
- s=0.9: +0.0023
- s=0.95: +0.0027

This supports the idea that CKA guidance may help, but it is not strong enough evidence by itself. Multiple seeds are needed before making a firm claim.

## Plot: `accuracy_vs_communication_cost.png`

### How It Was Calculated

The plotting code looks for these columns in order:

1. `communication_cost`
2. `active_params_transmitted`
3. `active_params`

The current logs do not contain true communication cost or transmitted-byte counts, so the plot uses `active_params` as a proxy.

- X-axis: active parameters
- Y-axis: test accuracy
- Each curve: one sparse method-sparsity combination

### What Each Section Means

- The title says the plot compares accuracy against communication cost.
- In the current run, the x-axis should be interpreted as active parameter count, not true communication bytes.
- Lower active parameters indicate a smaller sparse model.
- The y-axis shows test accuracy.

### Analysis

The active parameter counts are:

| Sparsity | Active Parameters |
|---:|---:|
| 0.5 | about 210,704 |
| 0.7 | about 126,422 |
| 0.8 | about 84,282 |
| 0.9 | about 42,141 |
| 0.95 | about 21,070 |

The plot should be read as an accuracy-versus-model-size proxy. It is not yet a full communication-cost analysis because it does not account for:

- number of clients
- number of rounds
- mask transmission cost
- gradient or topology update cost
- dense FedAvg communication bytes
- uplink and downlink separately

Still, it shows an important pattern: FedDST and CKA-FedDST maintain high accuracy at much lower active parameter counts than fixed Sparse FedAvg.

## Plot: `cka_feddst_layerwise_sparsity.png`

### How It Was Calculated

- Source columns:
  - `round`
  - `sparsity_conv1_weight`
  - `sparsity_conv2_weight`
  - `sparsity_fc1_weight`
  - `sparsity_fc2_weight`
- Only CKA-FedDST logs are used.
- X-axis: communication round
- Y-axis: layer-wise sparsity
- Each line: one layer at one base sparsity level

### What Each Section Means

- The title says the figure tracks how sparsity changes layer by layer.
- The x-axis shows federated rounds.
- The y-axis shows the fraction of inactive weights in a layer.
- The legend identifies both base sparsity and layer.

### Analysis

The total sparsity remains controlled and very close to each target:

| Target Sparsity | Final Total Sparsity |
|---:|---:|
| 0.5 | 0.5001 |
| 0.7 | 0.7001 |
| 0.8 | 0.8000 |
| 0.9 | 0.9000 |
| 0.95 | 0.9500 |

Final CKA-FedDST layer sparsities:

| Target | conv1 | conv2 | fc1 | fc2 |
|---:|---:|---:|---:|---:|
| 0.5 | 0.500 | 0.500 | 0.500 | 0.526 |
| 0.7 | 0.698 | 0.699 | 0.700 | 0.718 |
| 0.8 | 0.799 | 0.799 | 0.800 | 0.811 |
| 0.9 | 0.899 | 0.900 | 0.900 | 0.904 |
| 0.95 | 0.948 | 0.950 | 0.950 | 0.952 |

The layer-wise adjustments are modest. This is expected because `cka_strength` is 0.2, which is conservative. Also, `fc2` is no longer assigned a CKA-guided target, so its sparsity stays governed by normal FedDST active-count behavior rather than direct CKA adaptation.

## Plot: `cka_feddst_layerwise_cka.png`

### How It Was Calculated

- Source columns:
  - `round`
  - `cka_conv1`
  - `cka_conv2`
  - `cka_fc1`
- Only CKA-FedDST logs are used.
- CKA is computed every `cka_interval` rounds.
- In this experiment, `cka_interval` is 2.
- X-axis: communication round
- Y-axis: average pairwise client CKA
- Each line: one measured layer at one base sparsity level

### What Each Section Means

- The title says the figure tracks representation similarity over rounds.
- The x-axis shows communication rounds.
- The y-axis shows average pairwise CKA across clients.
- Higher CKA means clients have more similar representations for that layer.
- Lower CKA means client representations are more different.

### Analysis

Final CKA values at round 20:

| Sparsity | conv1 | conv2 | fc1 |
|---:|---:|---:|---:|
| 0.5 | 1.000 | 1.000 | 0.984 |
| 0.7 | 1.000 | 1.000 | 0.974 |
| 0.8 | 1.000 | 1.000 | 0.983 |
| 0.9 | 1.000 | 1.000 | 0.983 |
| 0.95 | 1.000 | 1.000 | 0.963 |

The convolutional layers have extremely high CKA, close to 1.0. This suggests early representations are highly shared across clients despite the non-IID split. The `fc1` layer has lower CKA, especially at 0.95 sparsity, which suggests later hidden representations are more sensitive to sparsity and client heterogeneity.

This behavior is consistent with the research idea: layers with more shared representation structure should be preserved more, while layers with less shared structure can absorb more sparsity. However, because CKA values are very high overall, the CKA signal is mild in this run.

## Overall Result Interpretation

The experiment supports three useful prototype-level conclusions:

1. Dense FedAvg remains the strongest upper baseline.
   - Final accuracy: 0.9124
   - Best accuracy: 0.9190

2. Dynamic sparse training is much more robust than fixed Sparse FedAvg at high sparsity.
   - At 0.95 sparsity, Sparse FedAvg collapses to 0.1135 final accuracy.
   - FedDST reaches 0.8003.
   - CKA-FedDST reaches 0.8031.

3. CKA-FedDST is competitive with FedDST and slightly better in most sparsity settings.
   - The advantage is small in this single-seed experiment.
   - The strongest claim currently supported is that CKA-FedDST is a reasonable and stable prototype, not that it decisively outperforms FedDST.

## Important Limitations

- These plots use a single seed, so they are not enough for a strong research claim.
- The communication-cost plot uses active parameters as a proxy, not true communication bytes.
- CKA guidance is conservative because `cka_strength` is 0.2.
- The experiment uses only MNIST and a small CNN.
- The non-IID setting is controlled by one alpha value, 0.3.
- More seeds, datasets, and model architectures are needed before drawing publishable conclusions.

## Recommended Next Analysis

1. Repeat the sweep over at least 3 seeds.
2. Add mean and standard deviation plots.
3. Add true communication-cost logging.
4. Sweep `cka_strength`, for example 0.1, 0.2, 0.5, and 1.0.
5. Test Fashion-MNIST before moving to CIFAR-10.
