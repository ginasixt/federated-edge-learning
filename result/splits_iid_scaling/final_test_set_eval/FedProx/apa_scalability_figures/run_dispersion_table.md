**Table X**

*Run-to-Run Dispersion in Threshold-Independent FedProx Test Performance*

| Metric | Mean SD across scaling points | SD range | Mean CV (%) | Min. CV (%) | Client at min. CV | Max. CV (%) | Client at max. CV | Max. relative single-run deviation (%) | Client at max. relative deviation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ROC-AUC | 0.0011 | 0.0002–0.0020 | 0.13 | 0.03 | 16 | 0.24 | 32 | 0.36 | 32 |
| AP | 0.0034 | 0.0010–0.0053 | 0.85 | 0.24 | 2 | 1.34 | 512 | 2.02 | 16,384 |
| Weighted loss | 0.0015 | 0.0005–0.0042 | 0.30 | 0.10 | 4 | 0.78 | 16,384 | 1.16 | 16,384 |

*Note. Statistics are based on five repeated runs at each of 14 scaling points. Mean SD is the arithmetic mean of the point-specific sample standard deviations calculated separately across the five runs at each scaling point; it is not a pooled standard deviation. The coefficient of variation (CV) is the point-specific sample standard deviation divided by the corresponding point-specific mean and expressed as a percentage. ROC-AUC, AP, and weighted loss are based on checkpoints selected using their corresponding validation metrics.*
