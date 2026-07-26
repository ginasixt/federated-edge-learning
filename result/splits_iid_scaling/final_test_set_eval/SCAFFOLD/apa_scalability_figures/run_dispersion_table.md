**Table X**

*Run-to-Run Dispersion in Threshold-Independent SCAFFOLD Test Performance*

| Metric | Mean SD across scaling points | SD range | Mean CV (%) | Min. CV (%) | Client at min. CV | Max. CV (%) | Client at max. CV | Max. relative single-run deviation (%) | Client at max. relative deviation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ROC-AUC | 0.0016 | 0.0006–0.0035 | 0.20 | 0.08 | 64 | 0.44 | 256 | 0.70 | 256 |
| AP | 0.0043 | 0.0015–0.0074 | 1.10 | 0.40 | 8,192 | 1.87 | 16 | 2.33 | 16 |
| Weighted loss | 0.0022 | 0.0011–0.0038 | 0.43 | 0.21 | 2,048 | 0.74 | 256 | 1.09 | 256 |

*Note. Statistics are based on five repeated runs at each of 14 scaling points. Mean SD is the arithmetic mean of the point-specific sample standard deviations calculated separately across the five runs at each scaling point; it is not a pooled standard deviation. The coefficient of variation (CV) is the point-specific sample standard deviation divided by the corresponding point-specific mean and expressed as a percentage. ROC-AUC, AP, and weighted loss are based on checkpoints selected using their corresponding validation metrics.*
