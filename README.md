# Credit risk and approval optimization

Two linked data-science problems on consumer-loan applications, plus a small optimization library built for the second one.

1. **Probability of default (PD).** A scorecard-style logistic regression and a gradient-boosting model, with time-stability
   screening of the features, WOE encoding and coarse classing, calibration and a one-shot holdout (May 2015).
2. **Approval policy optimization.** Choose approve / reject for each application to maximize expected profit under seven
   business constraints, solved as a binary mixed-integer program and then distilled into a small rule set (a 32-leaf
   decision tree, exported as JSON and SQL).

The optimization library is **PyDO** (Pythonic Decision Optimizer, package `pydo`, in `src/pydo`).

## Results at a glance

| | |
|---|---|
| PD model (LightGBM, calibrated) | Gini 0.701, AUC 0.851, KS 0.534 on the May-2015 holdout |
| Simple alternative (logistic + 1 interaction feature) | Gini 0.694 |
| Expected profit, approve everything | 11.98 bn VND |
| Expected profit, 32-rule policy (all 42 constraint rows satisfied) | 59.31 bn VND (97.35% of the account-level optimum) |
| Same policy on realized outcomes (back-test) | 58.28 bn VND |

The economics (interest, costs, recovery, default timing) are the synthetic assumptions of the exercise, not figures from a
real portfolio.

## Repository layout

```
.
├── notebooks/
│   ├── 01_pd_model.ipynb                  # Challenge 1: the PD model; writes results/pd_predictions.csv
│   ├── 02_portfolio_optimization.ipynb    # Challenge 2: the optimization; uses PyDO
│   └── functions/                         # profit functions written by notebook 02 and read by PyDO
├── src/pydo/                              # the PyDO library
├── tests/                                 # 43 unit tests of the library
├── results/
│   ├── optimal_policy.csv                 # the distilled policy: 32 rules + KPI and constraint summary
│   └── models/clx_approval_tree.{json,sql}  # the final policy as rules (JSON) and as a SQL CASE expression
├── docs/                                  # documentation (in Vietnamese): the two challenges and PyDO
├── data/                                  # input data goes here (not included, see data/README.md)
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

`results/pd_predictions.csv` and `results/constrained_output.csv` are account-level outputs. They are not committed;
running the notebooks re-creates them.

## Getting started

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt     # Linux/macOS: .venv/bin/python
```

This installs the pinned dependencies and PyDO in editable mode. Then:

1. Put the input data at `data/training_data.csv.gz` (see `data/README.md`; the file is not included).
2. Register a kernel and open the notebooks **from the `notebooks/` folder** (they use paths relative to it):
   ```bash
   .venv/Scripts/python.exe -m ipykernel install --user --name pydo-project
   ```
3. Run `01_pd_model.ipynb` first (about 5 minutes), then `02_portfolio_optimization.ipynb` (about 10 minutes).

Run the library tests with:

```bash
.venv/Scripts/python.exe -m unittest discover -s tests
```

## Documentation

The documents in `docs/` are written in Vietnamese:

| File | Content |
|---|---|
| [docs/README.md](docs/README.md) | Index, environment, reading order |
| [docs/challenge_1_pd_model.md](docs/challenge_1_pd_model.md) | The PD model: data audit, feature stability, modelling, validation |
| [docs/challenge_2_portfolio_optimization.md](docs/challenge_2_portfolio_optimization.md) | The optimization: economics, constraints, results, distillation, back-test |
| [docs/pydo.md](docs/pydo.md) | The PyDO library: concepts, rule language, API, tests, limitations |

## Limitations

- The profit model is synthetic; the profit level moves a lot with the recovery and default-timing assumptions, while the
  chosen policy moves little (2-4% of decisions).
- One month of data carries a PD, so the distilled tree has no out-of-time test.
- PyDO has been checked against brute-force enumeration and an independent constraint checker, but not against another
  optimization tool.

## License

MIT, see [LICENSE](LICENSE).
