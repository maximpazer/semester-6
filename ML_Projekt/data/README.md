# Data

## Feature dictionary

| Column | Type | Notes |
|--------|------|-------|
| `f00`–`f15` | float | 16 features, originally `N(0,1)`; some corrupted (see below) |
| `label` | 0/1 | target (training only); ~10 % positive |

For dataset `01111` the corruption pattern is:

| Feature(s) | Corruption | Modelling use |
|------------|-----------|---------------|
| `f11`, `f15` | MNAR missingness (~15 %, label-dependent) | **missing indicator = top signal** |
| `f09`, `f15` | heavy-tail outliers (kurtosis ≈ 17) | signed-log transform only |
| `f10`, `f13`, `f14` | linear scaling | weak residual signal (kept) |
| `f00`–`f08`, `f12` | untouched clean `N(0,1)` | `f12` = causal driver; `f00`–`f07` = noise |

Full mechanics: [../docs/2_data_understanding.md](../docs/2_data_understanding.md).

## What is in git vs. distributed separately

To keep every clone small, **no data files live in git** — only this README (the data
dictionary) is tracked. All CSVs are bundled in a single ZIP attached to the repo's
GitHub **Releases**. See [Getting the data](#getting-the-data) below.

| Path | In git? | Size | Where it comes from |
|------|---------|------|--------------------|
| `data/README.md` (this file) | ✅ yes | tiny | cloned with the repo |
| `toxic_data_01111.csv`, `toxic_exam_01111.csv` | ❌ no | ~3 MB | data ZIP — **required** to run the pipeline |
| `toxic_{data,exam}_{00000,00001,00010,00011,00100}.csv` | ❌ no | ~15 MB | data ZIP — peer sets for `compare_datasets.py` / `reverse_engineer.py` |
| `download/` | ❌ no | ~360 MB | data ZIP — bulk enriched dump (optional, only `diagnostics.py`) |

## The `download/` dump

`download/` holds four families, each spanning all 32 five-bit IDs (`00000`…`11111`):

- **`Toxic_Data/` · `Toxic_Exam/`** — our task family (8 000 / 2 000 rows, `f00`–`f15` + `label`).
- **`Lethal_Data/` · `Lethal_Exam/`** — a parallel, harder track (24 000 / 6 000 rows,
  `feat_00`–`feat_15` + `target`, **no** missingness/injections). **Not usable for the Toxic task.**

It is only needed to re-run [`../src/data_understanding/diagnostics.py`](../src/data_understanding/diagnostics.py)
(which reads `data/download/`). Every other script needs only the `01111` files.

## Getting the data

The datasets are **not** in Git history (too large; would bloat every clone). They are
distributed as a single ZIP attached to the repo's GitHub **Releases**. To set up locally:

1. Download the data ZIP (e.g. `ML_Projekt_data.zip`) from the repo's **Releases** page.
2. Extract it **into this `data/` folder**, so you end up with:

   ```
   data/
   ├── toxic_data_01111.csv      # required
   ├── toxic_exam_01111.csv      # required
   ├── toxic_*_000xx.csv         # peer sets (optional)
   └── download/                 # bulk enriched dump (optional)
   ```

3. Done — every script resolves paths relative to `data/`, so they now run unchanged.

**Minimum to reproduce the result:** just the `01111` pair. The peer sets and the
`download/` dump are only needed for the data-understanding / diagnostics scripts.

> Conclusion from [../docs/5_evaluation.md](../docs/5_evaluation.md): the enriched data is
> great for *understanding* the generator but does **not** raise the score — `01111` is already
> near its ceiling. So the missing dump does not affect the result.
