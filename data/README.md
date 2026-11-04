# Datasets

The datasets are not part of this repository. Download them from their
original sources and place the files directly in this folder. Please follow
each provider's license and terms of use.

| Dataset | Files | Source |
|---|---|---|
| CreditCard (Dal Pozzolo et al., 2015) | `creditcard.csv` | Kaggle: `mlg-ulb/creditcardfraud` |
| Bank Account Fraud (Jesus et al., 2022) | `Base.csv`, `Variant I.csv`, `Variant II.csv`, `Variant III.csv`, `Variant IV.csv`, `Variant V.csv` | Kaggle: `sgpjesus/bank-account-fraud-dataset-neurips-2022` |

The BAF file names contain a space and must be kept exactly as above.
Experiments 19 and 21 store an encoded copy of the BAF files in `data/cache/`
the first time they run.
