# Data

`heart.csv` is the Cleveland subset of the UCI Heart Disease dataset: 303 patient
records, 13 clinical measurements and one binary outcome column.

## Attribution

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
> *Heart Disease* [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C52P4X

**Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).**
Redistribution is permitted with attribution, which is why the file is included
here rather than linked.

Source page: https://archive.ics.uci.edu/dataset/45/heart+disease

## Provenance caveat

This file is the widely circulated redistribution of the Cleveland data, not a
byte-for-byte copy of UCI's `processed.cleveland.data`. Two differences were
confirmed while preparing this repository:

1. It contains **one exact duplicate row**, removed in the analysis, leaving 302
   patients.
2. Its class split is **165 / 138**, against UCI's documented **164 / 139** for
   `num = 0` and `num > 0`. One record differs.

## The outcome column

The column is named `target`, and its direction is **inverted relative to UCI's
original `num`**. See the "Outcome label" section of the top-level README for the
evidence. In short, `target = 1` corresponds to UCI's `num = 0`, meaning
**disease absent**.

Two columns carry codes outside their documented ranges, treated as missing data
rather than as categories:

| Column | Documented range | Out-of-range code | Affected rows |
|---|---|---|---|
| `ca` | 0 to 3 | `4` | 4 |
| `thal` | 1 to 3 | `0` | 2 |
