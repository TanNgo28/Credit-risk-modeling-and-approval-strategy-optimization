# Data

The input data is **not included in this repository**.

To run the notebooks, place the training file at:

```
data/training_data.csv.gz
```

The file is a gzip-compressed CSV with one row per loan application (about 200,000 rows and 61 columns, applications from
20 June 2014 to 31 May 2015). The columns are described in `docs/challenge_1_pd_model.md` (section 3): an application id
(`ID_APPLICATION`), a timestamp (`TIME`), the binary target (`TARGET`, 1 = default), numerical and categorical features,
and the contract columns used only by Challenge 2.

Everything in `data/` except this file is ignored by git (see `.gitignore`).
