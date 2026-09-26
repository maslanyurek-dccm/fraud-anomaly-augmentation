# Datasets

The datasets are **not** included in this repository. Download them from their
original sources and place the files directly in this `data/` folder.
Please check and respect each provider's licence and terms of use.

| Dataset | Files expected here | Source |
|---|---|---|
| CreditCard (Dal Pozzolo et al., 2015) | `creditcard.csv` | Kaggle: `mlg-ulb/creditcardfraud` |
| Bank Account Fraud (BAF) suite (Jesus et al., NeurIPS 2022) | `Base.csv`, `Variant I.csv`, `Variant II.csv`, `Variant III.csv`, `Variant IV.csv`, `Variant V.csv` | Kaggle: `sgpjesus/bank-account-fraud-dataset-neurips-2022` |

File names must match exactly (the BAF variant files contain a space).

Expected sizes after the temporal splits used in the article:

- CreditCard: 284,807 rows; train 170,884 (360 frauds), validation 56,961 (57), test 56,962 (75).
- BAF Base Month 7 test section: 96,843 rows (1,428 frauds).
