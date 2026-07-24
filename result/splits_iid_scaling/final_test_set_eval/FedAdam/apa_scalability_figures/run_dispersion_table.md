**Table X**

*Run-to-Run Dispersion in Threshold-Independent FedAdam Test Performance*

| Metric | Mean SD across scaling points | SD range | Mean CV (%) | Min. CV (%) | Client at min. CV | Max. CV (%) | Client at max. CV | Max. relative single-run deviation (%) | Client at max. relative deviation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ROC-AUC | 0.0007 | 0.0003–0.0014 | 0.08 | 0.03 | 8,192 | 0.17 | 16,384 | 0.31 | 16,384 |
| AP | 0.0025 | 0.0002–0.0046 | 0.60 | 0.05 | 2 | 1.10 | 32 | 1.80 | 16,384 |
| Weighted loss | 0.0014 | 0.0004–0.0075 | 0.27 | 0.08 | 4 | 1.32 | 32,768 | 1.96 | 32,768 |

*Note. Statistics are based on five repeated runs at each of 15 scaling points. Mean SD is the arithmetic mean of the 15 sample standard deviations calculated separately across the five runs at each scaling point; it is not a pooled standard deviation. The coefficient of variation (CV) is the point-specific sample standard deviation divided by the corresponding point-specific mean and expressed as a percentage. ROC-AUC, AP, and weighted loss are based on checkpoints selected using their corresponding validation metrics.*
