# Fashion-MNIST Experiment Analysis

## Scope

This report summarizes the Fashion-MNIST experiments for CKA-guided dynamic sparse federated learning.

- Main comparison: `results/averaged/multiseed/fashion_mnist_all_20260517_180855`
- CKA-strength sweep: `results/averaged/cka_strength_sweep/fashion_mnist_all_20260517_180855`
- Seeds: `7, 13, 21, 42, 100`
- Completed runs: 80 multiseed runs and 125 CKA-strength sweep runs

## Main Results

Dense FedAvg reached `0.7795 +/- 0.0172` final test accuracy.

| Sparsity | Sparse FedAvg | FedDST | CKA-FedDST |
| --- | ---: | ---: | ---: |
| 0.50 | 0.7398 | 0.7369 | 0.7376 |
| 0.70 | 0.7194 | 0.7226 | 0.7239 |
| 0.80 | 0.7008 | 0.7125 | 0.7121 |
| 0.90 | 0.6057 | 0.7038 | 0.7085 |
| 0.95 | 0.1000 | 0.6557 | 0.6407 |

## CKA-FedDST Versus FedDST

| Sparsity | Final Accuracy Delta |
| --- | ---: |
| 0.50 | +0.0007 |
| 0.70 | +0.0013 |
| 0.80 | -0.0004 |
| 0.90 | +0.0047 |
| 0.95 | -0.0149 |

CKA-FedDST is nearly identical to FedDST through most sparsity levels. The largest positive difference appears at `0.90` sparsity, but it is still small.

## CKA Behavior

Mean layer-wise CKA values were highly saturated:

- `conv1`: 1.0000
- `conv2`: 0.9999
- `fc1`: 0.9745

The CKA signal remains very compressed near 1.0, which limits its usefulness for differentiating layers.

## CKA Strength Sweep

Across the CKA-strength sweep, final accuracy changed minimally:

| Sparsity | Final Accuracy Range |
| --- | ---: |
| 0.50 | 0.7376-0.7376 |
| 0.70 | 0.7234-0.7254 |
| 0.80 | 0.7121-0.7154 |
| 0.90 | 0.7076-0.7086 |
| 0.95 | 0.6407-0.6407 |

The CKA-strength parameter does not meaningfully change Fashion-MNIST performance in the current setup.

## Conclusion

Fashion-MNIST is more difficult than MNIST, but the same pattern holds: dynamic sparse training is much stronger than static Sparse FedAvg at high sparsity, while CKA guidance adds little beyond FedDST. The main limitation is still saturated CKA.
