# CKA-Guided FedDST Research Analysis Report

Date: 2026-05-17

Project: CKA-Guided Dynamic Sparse Training for Federated Learning under Non-IID Data

This report summarizes the current experimental evidence from the existing logs, averaged CSV files, and generated plots. It does not introduce new experiments or new analysis scripts. The conclusions are based on the already generated multiseed and CKA-strength-sweep results.

## 1. Executive Summary

The current experiments provide a useful and coherent 1-day research prototype result. The strongest finding is that dynamic sparse training is substantially more robust than fixed sparse training under high sparsity.

The current results support this claim:

> Dynamic sparse training, represented by FedDST, is much more effective than fixed Sparse FedAvg at high sparsity under the current MNIST non-IID setup.

The current results do not yet strongly support this claim:

> CKA-guided layer-wise sparsity improves FedDST.

CKA-FedDST is implemented and works end-to-end, but its accuracy is very close to normal FedDST. In several high-sparsity settings, CKA-FedDST is slightly worse than FedDST. The most likely reason is that the CKA signal saturates: `conv1` and `conv2` have CKA close to `1.0`, and `fc1` is also very high. Because the layers already look highly similar across clients, CKA-guided sparsity has little useful variation to exploit.

Overall verdict:

**The code and experiments are strong enough for an MVP research prototype, but the current results are not enough to claim that CKA guidance improves FedDST.**

## 2. Evidence Analyzed

The analysis used the current completed suite:

```text
all_20260516_024344
```

Raw log folders:

```text
results/logs/multiseed/all_20260516_024344/
results/logs/cka_strength_sweep/all_20260516_024344/
```

Averaged result folders:

```text
results/averaged/multiseed/all_20260516_024344/
results/averaged/cka_strength_sweep/all_20260516_024344/
```

Plot folders:

```text
results/plots/multiseed/all_20260516_024344/
results/plots/cka_strength_sweep/all_20260516_024344/
```

The aggregation manifests report:

| Suite | Raw rows | Unique runs | Seeds |
|---|---:|---:|---|
| multiseed | 1600 | 80 | 7, 13, 21, 42, 100 |
| cka_strength | 2500 | 125 | 7, 13, 21, 42, 100 |

The multiseed suite contains:

```text
5 seeds x 16 runs = 80 runs
```

The CKA-strength sweep contains:

```text
5 seeds x 5 sparsities x 5 CKA strengths = 125 runs
```

## 3. Experimental Setting

Dataset:

```text
MNIST
```

Federated setup:

```text
num_clients = 5
alpha = 0.3
rounds = 20
local_epochs = 1
```

Compared methods:

1. FedAvg
2. Sparse FedAvg
3. FedDST
4. CKA-FedDST

Sparse methods were evaluated at:

```text
sparsity = 0.5, 0.7, 0.8, 0.9, 0.95
```

CKA-strength values tested:

```text
cka_strength = 0.2, 0.5, 0.8, 0.9, 1.0
```

The multiseed baseline uses CKA-FedDST with:

```text
cka_strength = 0.2
```

## 4. Main Multiseed Accuracy Results

### 4.1 Final Test Accuracy

| Method | Sparsity | Final accuracy mean | Std |
|---|---:|---:|---:|
| FedAvg | dense | 0.9415 | 0.0167 |
| Sparse FedAvg | 0.5 | 0.9087 | 0.0224 |
| Sparse FedAvg | 0.7 | 0.8838 | 0.0234 |
| Sparse FedAvg | 0.8 | 0.8647 | 0.0254 |
| Sparse FedAvg | 0.9 | 0.5358 | 0.2008 |
| Sparse FedAvg | 0.95 | 0.1064 | 0.0065 |
| FedDST | 0.5 | 0.9124 | 0.0201 |
| FedDST | 0.7 | 0.8959 | 0.0233 |
| FedDST | 0.8 | 0.8857 | 0.0194 |
| FedDST | 0.9 | 0.8610 | 0.0266 |
| FedDST | 0.95 | 0.7223 | 0.0663 |
| CKA-FedDST | 0.5 | 0.9130 | 0.0197 |
| CKA-FedDST | 0.7 | 0.8968 | 0.0241 |
| CKA-FedDST | 0.8 | 0.8828 | 0.0214 |
| CKA-FedDST | 0.9 | 0.8570 | 0.0271 |
| CKA-FedDST | 0.95 | 0.6908 | 0.1547 |

### 4.2 Best Test Accuracy

| Method | Sparsity | Best accuracy mean | Std |
|---|---:|---:|---:|
| FedAvg | dense | 0.9433 | 0.0146 |
| Sparse FedAvg | 0.5 | 0.9116 | 0.0198 |
| Sparse FedAvg | 0.7 | 0.8864 | 0.0205 |
| Sparse FedAvg | 0.8 | 0.8666 | 0.0226 |
| Sparse FedAvg | 0.9 | 0.5379 | 0.1971 |
| Sparse FedAvg | 0.95 | 0.1064 | 0.0065 |
| FedDST | 0.5 | 0.9140 | 0.0186 |
| FedDST | 0.7 | 0.9015 | 0.0171 |
| FedDST | 0.8 | 0.8882 | 0.0163 |
| FedDST | 0.9 | 0.8659 | 0.0216 |
| FedDST | 0.95 | 0.7272 | 0.0682 |
| CKA-FedDST | 0.5 | 0.9142 | 0.0186 |
| CKA-FedDST | 0.7 | 0.9032 | 0.0168 |
| CKA-FedDST | 0.8 | 0.8878 | 0.0158 |
| CKA-FedDST | 0.9 | 0.8630 | 0.0215 |
| CKA-FedDST | 0.95 | 0.7040 | 0.1565 |

## 5. Interpretation of Main Results

### 5.1 Dense FedAvg Remains the Strongest Accuracy Baseline

Dense FedAvg reaches final accuracy:

```text
0.9415 +/- 0.0167
```

None of the sparse methods match dense FedAvg. This is expected because dense FedAvg has the full parameter budget.

The sparse methods should therefore not be judged only by raw accuracy. They should also be judged by accuracy retained at reduced active parameter count and reduced communication proxy.

### 5.2 Dynamic Sparse Training Strongly Beats Fixed Sparse FedAvg at High Sparsity

Sparse FedAvg performs reasonably at moderate sparsity, but it collapses at high sparsity:

```text
Sparse FedAvg s=0.9  final accuracy = 0.5358
Sparse FedAvg s=0.95 final accuracy = 0.1064
```

FedDST remains much stronger:

```text
FedDST s=0.9  final accuracy = 0.8610
FedDST s=0.95 final accuracy = 0.7223
```

This is the clearest positive result in the project. It suggests that dynamic pruning and regrowth are critical when the model is highly sparse.

### 5.3 CKA-FedDST Is Competitive but Not Better Than FedDST

CKA-FedDST slightly improves over FedDST at lower sparsity:

| Sparsity | CKA-FedDST | FedDST | Difference |
|---:|---:|---:|---:|
| 0.5 | 0.9130 | 0.9124 | +0.0006 |
| 0.7 | 0.8968 | 0.8959 | +0.0009 |

But it is worse at higher sparsity:

| Sparsity | CKA-FedDST | FedDST | Difference |
|---:|---:|---:|---:|
| 0.8 | 0.8828 | 0.8857 | -0.0029 |
| 0.9 | 0.8570 | 0.8610 | -0.0040 |
| 0.95 | 0.6908 | 0.7223 | -0.0315 |

The differences at `s=0.5` and `s=0.7` are too small to be meaningful. The high-sparsity loss at `s=0.95` is more concerning because it is larger and accompanied by high variance.

## 6. CKA-Strength Sweep

### 6.1 Final Accuracy by CKA Strength

Average final accuracy across all sparsities:

| CKA strength | Mean final accuracy |
|---:|---:|
| 0.2 | 0.8481 |
| 0.5 | 0.8482 |
| 0.8 | 0.8484 |
| 0.9 | 0.8491 |
| 1.0 | 0.8496 |

The strongest setting is `cka_strength = 1.0`, but the gain over `0.2` is only about:

```text
0.0015 absolute accuracy
```

This is much smaller than the observed seed-to-seed standard deviation, so it should not be treated as a strong research finding.

### 6.2 Best Accuracy by CKA Strength

Average best accuracy across all sparsities:

| CKA strength | Mean best accuracy |
|---:|---:|
| 0.2 | 0.8544 |
| 0.5 | 0.8545 |
| 0.8 | 0.8551 |
| 0.9 | 0.8553 |
| 1.0 | 0.8556 |

Again, stronger CKA guidance is slightly better numerically, but the effect is very small.

### 6.3 Best Strength by Sparsity

For final accuracy:

| Sparsity | Best CKA strength | Final accuracy |
|---:|---:|---:|
| 0.5 | 0.2 | 0.9130 |
| 0.7 | 1.0 | 0.8976 |
| 0.8 | 1.0 | 0.8866 |
| 0.9 | 1.0 | 0.8599 |
| 0.95 | 0.2 | 0.6908 |

For best accuracy:

| Sparsity | Best CKA strength | Best accuracy |
|---:|---:|---:|
| 0.5 | 0.2 | 0.9142 |
| 0.7 | 0.2 | 0.9032 |
| 0.8 | 1.0 | 0.8903 |
| 0.9 | 0.9 | 0.8668 |
| 0.95 | 0.2 | 0.7040 |

The moderate high-sparsity region, especially `s=0.8` and `s=0.9`, appears to benefit slightly from stronger CKA guidance. However, the gain is still very small.

## 7. CKA Signal Analysis

The final layer-wise CKA values are extremely high.

For CKA-FedDST with `cka_strength = 0.2`, final round CKA is:

| Sparsity | conv1 | conv2 | fc1 |
|---:|---:|---:|---:|
| 0.5 | 1.0000 | 1.0000 | 0.9774 |
| 0.7 | 1.0000 | 1.0000 | 0.9749 |
| 0.8 | 1.0000 | 1.0000 | 0.9710 |
| 0.9 | 1.0000 | 1.0000 | 0.9732 |
| 0.95 | 1.0000 | 1.0000 | 0.9654 |

Across the CKA-strength sweep, final CKA remains approximately:

```text
conv1 = 1.0000
conv2 = 1.0000
fc1   = about 0.972
```

This is a key scientific observation.

The proposed method depends on CKA being informative enough to distinguish more shared layers from more client-specific layers. In the current MNIST/SmallCNN setup, CKA is nearly saturated. That means the method does not receive a strong layer-ranking signal.

As a result, CKA-guided sparsity behaves very similarly to normal FedDST.

## 8. Layer-Wise Sparsity Analysis

Total sparsity is well controlled. Final total sparsity is essentially equal to the requested value:

| Target sparsity | Observed total sparsity |
|---:|---:|
| 0.5 | 0.5000 |
| 0.7 | 0.7000 |
| 0.8 | 0.8000 |
| 0.9 | 0.9000 |
| 0.95 | 0.9500 |

Active parameter counts also scale correctly:

| Sparsity | Active parameters |
|---:|---:|
| 0.5 | about 210,704 |
| 0.7 | about 126,422 |
| 0.8 | about 84,282 |
| 0.9 | about 42,141 |
| 0.95 | about 21,070 |

Total trainable sparse parameter budget:

```text
421,408 parameters
```

This confirms that sparsity accounting is working.

However, the actual layer-wise CKA-guided sparsity shifts are small. For example, at `s=0.8` and `cka_strength=1.0`, final layer sparsities are approximately:

| Layer | Final sparsity |
|---|---:|
| conv1 | 0.7951 |
| conv2 | 0.7945 |
| fc1 | 0.8003 |
| fc2 | 0.8008 |

This is very close to uniform `0.8` sparsity. The CKA guidance is technically active, but practically weak in this setting.

## 9. High-Sparsity Stability

High sparsity increases seed variance.

At `s=0.95`, CKA-FedDST final accuracy varies substantially:

| Seed | Final accuracy |
|---:|---:|
| 7 | 0.6818 |
| 13 | 0.4334 |
| 21 | 0.7517 |
| 42 | 0.8412 |
| 100 | 0.7460 |

FedDST at `s=0.95` is more stable:

| Seed | Final accuracy |
|---:|---:|
| 7 | 0.7315 |
| 13 | 0.6783 |
| 21 | 0.6480 |
| 42 | 0.8221 |
| 100 | 0.7314 |

Sparse FedAvg at `s=0.95` is consistently near random:

| Seed | Final accuracy |
|---:|---:|
| 7 | 0.1032 |
| 13 | 0.1135 |
| 21 | 0.1010 |
| 42 | 0.1135 |
| 100 | 0.1010 |

This suggests that fixed sparse masks fail reliably at extreme sparsity, while dynamic sparse masks remain viable but can become unstable.

## 10. Communication and Parameter Efficiency Interpretation

Because sparse methods use fewer active parameters, the comparison should not be framed only as dense accuracy versus sparse accuracy.

The most interesting operating region is:

```text
sparsity = 0.8 to 0.9
```

At `s=0.8`, FedDST gets:

```text
0.8857 final accuracy with about 84k active parameters
```

At `s=0.9`, FedDST gets:

```text
0.8610 final accuracy with about 42k active parameters
```

Compared with dense FedAvg:

```text
0.9415 final accuracy with about 421k trainable sparse-budget parameters
```

This is a useful accuracy versus sparsity tradeoff. FedDST retains much of the dense model accuracy while using far fewer active weights.

The current communication-cost metric should still be described carefully. It uses active parameters as a proxy, not exact transmitted bytes.

## 11. Plot-Level Observations

### 11.1 Final Accuracy vs Sparsity

The multiseed final-accuracy plot clearly shows three regimes:

1. Moderate sparsity, `s=0.5` to `s=0.8`:
   - All sparse methods train reasonably.
   - FedDST and CKA-FedDST are slightly better than Sparse FedAvg.

2. High sparsity, `s=0.9`:
   - Sparse FedAvg degrades sharply.
   - FedDST and CKA-FedDST remain strong.

3. Extreme sparsity, `s=0.95`:
   - Sparse FedAvg collapses to random performance.
   - FedDST remains usable.
   - CKA-FedDST remains usable but is more variable and worse than FedDST.

### 11.2 Accuracy vs Rounds

The accuracy curves show learning progress for all viable methods. Dense FedAvg learns fastest and reaches the highest accuracy.

Sparse FedAvg at `s=0.9` and `s=0.95` learns very slowly or fails.

FedDST and CKA-FedDST continue improving even at high sparsity.

There is a noticeable temporary dip around round 15 in some curves. Raw logs show that this is caused by specific seeds temporarily collapsing and then recovering. This means final and best accuracy are more reliable for summary comparison than interpreting every single intermediate point.

### 11.3 CKA Strength Plots

The CKA-strength curves are nearly overlapping. This visually confirms the numeric conclusion: changing `cka_strength` currently has only a small effect.

### 11.4 Layer-Wise CKA Plot

The CKA plot shows that CKA is very high throughout training, especially for `conv1` and `conv2`. This suggests that the representation similarity measure has little discriminative power in this MNIST/SmallCNN setting.

### 11.5 Layer-Wise Sparsity Plot

The layer-wise sparsity plot confirms that sparsity is controlled. However, it is visually crowded. For final presentation, it would be better to split this plot by sparsity level or show only selected sparsity levels such as `0.8` and `0.9`.

## 12. Scientific Interpretation

The original research idea is:

> Preserve layers that are shared across clients and sparsify more client-specific layers using CKA as a layer-wise similarity signal.

This is scientifically reasonable, but the current experiment does not create a strong enough test of the idea. The reason is that CKA does not vary much across layers in the current setup.

In this experiment:

- Early convolution layers are almost perfectly aligned across clients.
- The `fc1` representation is also highly aligned.
- The classifier layer `fc2` is not directly CKA-adapted.
- The resulting sparsity targets remain close to the base sparsity.

Therefore, the method is meaningfully implemented, but the dataset/model combination does not expose the expected advantage.

## 13. Limitations of the Current Evidence

Important limitations:

1. The dataset is only MNIST.
2. The model is a small CNN.
3. The task may be too easy for CKA-guided layer adaptation to matter.
4. CKA values saturate, especially in `conv1` and `conv2`.
5. The CKA-guided sparsity shifts are small.
6. `fc2.weight` is excluded from direct CKA-guided adaptation.
7. Communication cost is an active-parameter proxy, not exact communication bytes.
8. Only one alpha value is used:

```text
alpha = 0.3
```

9. There are only 5 clients.
10. The number of communication rounds is still modest:

```text
rounds = 20
```

11. High sparsity, especially `s=0.95`, has high variance across seeds.

## 14. Research Claims Supported by Current Results

The current results support the following claims:

1. FedDST is much more robust than fixed Sparse FedAvg at high sparsity.
2. Fixed Sparse FedAvg can fail catastrophically at extreme sparsity.
3. Dynamic sparse training preserves useful accuracy even with a much smaller active parameter budget.
4. The sparsity implementation maintains the target sparsity accurately.
5. CKA-FedDST runs end-to-end and produces stable logs, CKA values, and layer-wise target sparsities.

## 15. Research Claims Not Yet Supported

The current results do not yet support the following stronger claims:

1. CKA-FedDST improves over FedDST.
2. CKA-guided layer-wise sparsity is consistently beneficial.
3. Higher `cka_strength` meaningfully improves performance.
4. CKA guidance improves communication efficiency beyond standard FedDST.
5. The proposed method generalizes beyond MNIST and SmallCNN.

## 16. Recommended Wording for a Report

A careful research report should say:

> In the MNIST non-IID prototype, dynamic sparse training substantially improves over fixed sparse FedAvg at high sparsity. CKA-guided FedDST is competitive with FedDST, but does not provide a clear improvement. The likely reason is that CKA similarities are saturated across the measured layers, so the CKA-guided sparsity allocation remains close to uniform. These results validate the implementation and motivate testing the method on harder datasets and deeper architectures where layer-wise representation divergence is larger.

Avoid saying:

> CKA-guided FedDST outperforms FedDST.

The evidence does not currently support that.

## 17. Recommended Next Experiments

### Highest Priority

Run the same comparison on a harder dataset:

1. Fashion-MNIST
2. CIFAR-10

The goal is to create a setting where client representations diverge more and CKA has a stronger layer-wise signal.

### Model Architecture

Test a deeper model:

1. deeper CNN
2. ResNet-style model
3. CNN with more representation layers

The current SmallCNN has only three CKA-tracked layers:

```text
conv1, conv2, fc1
```

That may be too shallow for layer-wise CKA-guided sparsity to show a strong effect.

### Non-IID Strength

Run an alpha sweep:

```text
alpha = 0.1, 0.3, 0.5, 1.0
```

The method may be more useful when non-IID heterogeneity is stronger.

### More Clients

Try:

```text
num_clients = 10, 20
```

More clients may increase representation diversity and make CKA more informative.

### Communication Cost

Add a more faithful communication-cost metric:

```text
number of transmitted nonzero weights
mask transmission cost
index overhead
full byte estimate
```

This would make sparse methods easier to evaluate as communication-efficient FL methods.

## 18. Final Verdict

The project currently has a strong baseline result:

> FedDST is clearly better than fixed Sparse FedAvg under high sparsity.

The proposed method is implemented and testable:

> CKA-FedDST runs correctly, maintains sparsity, logs CKA, and supports CKA-strength sweeps.

But the proposed method is not yet empirically validated:

> CKA-FedDST does not clearly improve over FedDST in the current MNIST/SmallCNN setup.

The most important next step is:

> Move to a harder dataset or deeper architecture where layer-wise CKA differences are not saturated.

That is the best path toward determining whether CKA-guided layer-wise sparsity has a real advantage.
