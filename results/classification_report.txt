Classification Report
=====================
Date: [Current Date]
Training Data: Biblical (5,823) vs Rabbinic (Balanced undersampled to 5,823)
Total Training Samples: 11,646

10-Fold Cross Validation Results (Target: Biblical)
---------------------------------------------------
Accuracy:  0.8940 (+/- 0.0151)
Precision: 0.8786 (+/- 0.0285)
Recall:    0.9146 (+/- 0.0191)
F1 Score:  0.8961 (+/- 0.0136)

Modern Hebrew Classification Results
------------------------------------
(Percentage of sentences classified as Biblical vs Rabbinic)

| Genre (Sub-Corpus) | Biblical (%) | Rabbinic (%) |
|-------------------|--------------|--------------|
| Blogs             | 11.04%       | 88.96%       |
| Medical           | 19.74%       | 80.26%       |
| News              | 21.35%       | 78.65%       |
| Tapuz (Forums)    | 14.34%       | 85.66%       |

Raw Counts:
-----------
Blogs:   Biblical=53,574, Rabbinic=431,495
Medical: Biblical=7,600,  Rabbinic=30,901
News:    Biblical=10,412, Rabbinic=38,356
Tapuz:   Biblical=17,876, Rabbinic=106,762

