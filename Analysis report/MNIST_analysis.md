# MNIST Experiment Analysis

## Scope

This report summarizes the MNIST experiments for CKA-guided dynamic sparse federated learning.

- Main comparison: `results/averaged/multiseed/mnist_all_20260516_024344`
- CKA-strength sweep: `results/averaged/cka_strength_sweep/mnist_all_20260516_024344`
- Seeds: `7, 13, 21, 42, 100`
- Completed runs: 80 multiseed runs and 125 CKA-strength sweep runs

## Main Results

Dense FedAvg reached `0.9415 +/- 0.0167` final test accuracy.

| Sparsity | Sparse FedAvg | FedDST | CKA-FedDST |
| --- | ---: | ---: | ---: |
| 0.50 | 0.9087 | 0.9124 | 0.9130 |
| 0.70 | 0.8838 | 0.8959 | 0.8968 |
| 0.80 | 0.8647 | 0.8857 | 0.8828 |
| 0.90 | 0.5358 | 0.8610 | 0.8570 |
| 0.95 | 0.1064 | 0.7223 | 0.6908 |

## CKA-FedDST Versus FedDST

| Sparsity | Final Accuracy Delta |
| --- | ---: |
| 0.50 | +0.0006 |
| 0.70 | +0.0009 |
| 0.80 | -0.0029 |
| 0.90 | -0.0039 |
| 0.95 | -0.0314 |

CKA-FedDST is effectively tied with FedDST at moderate sparsity and worse at extreme sparsity.

## CKA Behavior

Mean layer-wise CKA values were very high:

- `conv1`: 1.0000
- `conv2`: 0.9997
- `fc1`: 0.9593

This indicates strong CKA saturation. Because the layer similarities are already close to one another, CKA provides only a weak signal for layer-wise sparsity allocation.

## CKA Strength Sweep

Across the CKA-strength sweep, final accuracy varied only slightly:

| Sparsity | Final Accuracy Range |
| --- | ---: |
| 0.50 | 0.9130-0.9130 |
| 0.70 | 0.8966-0.8976 |
| 0.80 | 0.8828-0.8866 |
| 0.90 | 0.8568-0.8599 |
| 0.95 | 0.6908-0.6908 |

Changing `cka_strength` had little practical effect on MNIST.

## Conclusion

MNIST is too easy for the current CKA-guided mechanism to show a strong advantage. FedDST and CKA-FedDST both preserve good accuracy under sparsity, but CKA saturation limits the benefit of representation-guided sparsity allocation.
