# CIFAR-10 Experiment Analysis

## Scope

This report summarizes the CIFAR-10 experiments for CKA-guided dynamic sparse federated learning.

- Main comparison: `results/averaged/multiseed/cifar10`
- CKA-strength sweep: `results/averaged/cka_strength_sweep/cifar10`
- Seeds: `7, 13, 21, 42, 100`
- Completed runs: 80 multiseed runs and 125 CKA-strength sweep runs

## Main Results

Dense FedAvg reached `0.4490 +/- 0.0189` final test accuracy.

| Sparsity | Sparse FedAvg | FedDST | CKA-FedDST |
| --- | ---: | ---: | ---: |
| 0.50 | 0.3221 | 0.3202 | 0.3210 |
| 0.70 | 0.2439 | 0.2388 | 0.2394 |
| 0.80 | 0.1854 | 0.2021 | 0.2003 |
| 0.90 | 0.1000 | 0.1138 | 0.1221 |
| 0.95 | 0.1000 | 0.1000 | 0.1000 |

## CKA-FedDST Versus FedDST

| Sparsity | Final Accuracy Delta |
| --- | ---: |
| 0.50 | +0.0008 |
| 0.70 | +0.0006 |
| 0.80 | -0.0018 |
| 0.90 | +0.0083 |
| 0.95 | +0.0000 |

CKA-FedDST is essentially tied with FedDST. The largest gain occurs at `0.90` sparsity, but the absolute accuracy is close to random CIFAR-10 performance.

## CKA Behavior

Mean layer-wise CKA values remained saturated even on CIFAR-10:

- `conv1`: 0.9999
- `conv2`: 0.9993
- `conv3`: 0.9987
- `fc1`: 0.9827

CIFAR-10 is harder than MNIST and Fashion-MNIST, but the current CKA measurement still does not produce a strong layer-differentiating signal.

## CKA Strength Sweep

Across the CKA-strength sweep, final accuracy was mostly insensitive to `cka_strength`:

| Sparsity | Final Accuracy Range |
| --- | ---: |
| 0.50 | 0.3210-0.3210 |
| 0.70 | 0.2394-0.2413 |
| 0.80 | 0.1993-0.2011 |
| 0.90 | 0.1198-0.1291 |
| 0.95 | 0.1000-0.1000 |

The only visible sensitivity appears at `0.90` sparsity, but the result is too low in absolute accuracy to support a strong claim.

## Conclusion

CIFAR-10 confirms that the current sparse methods struggle on harder visual data, especially above `0.80` sparsity. More importantly, CKA still saturates, so the next research step should focus on improving the representation signal before running larger CIFAR sweeps.
