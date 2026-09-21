# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

A research framework for **quantifying the (dis)similarity between two NetFlow datasets** — in
particular, for evaluating how close *synthetic* (generated) NetFlow traffic is to *real* NetFlow
traffic. It was built to benchmark generative models (e.g. GPT-2-based and WGAN-based flow
generators) against real network intrusion-detection datasets.

Given two datasets `ds1` and `ds2`, the framework computes a battery of statistical, correlation,
and machine-learning-based metrics, aggregates them into two headline scores — a **Data
Dissimilarity Score** and a **Domain Dissimilarity Score** — and produces plots. For synthetic
data it additionally runs a **syntax check** that counts how many generated flows are invalid
NetFlow records.

There is no test suite, package config, or entry-point console script — this is a collection of
command-line scripts driven by a shared `src/` library, run in a fixed pipeline order.

## Data format

Input is CSV in the **Queensland binary NetFlow format** (nProbe NetFlow v9/IPFIX export, as used
by the [Queensland NIDS datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)). Columns:

```
IPV4_SRC_ADDR, L4_SRC_PORT, IPV4_DST_ADDR, L4_DST_PORT, PROTOCOL, L7_PROTO,
IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS, TCP_FLAGS, FLOW_DURATION_MILLISECONDS, Label, Attack
```

- `Label` is binary (0 = benign, 1 = attack). `Attack` and `L7_PROTO` are dropped by the framework.
- Samples are typically **10,000 rows** each (see `create_samples_cmd.py`, `nrows=10000` in syntax check).
- Datasets referenced throughout: `NF-UNSW-NB15`, `NF-ToN-IoT`, `NF-CSE-CIC-IDS2018`.

## Metrics computed (see `src/data_metrics.py`)

Attribute distribution distances (per attribute, then averaged):
- **Jensen-Shannon divergence (JS)** and **Wasserstein distance (WD)**. Categorical attributes use
  value-count proportions; numerical attributes use Gaussian KDE-sampled PDFs.

Correlation structure differences (MAE between the two datasets' correlation matrices):
- **Pearson** (numeric×numeric), **Correlation Ratio / eta** (categorical×numeric), **Theil's U**
  (categorical×categorical).

ML-based metrics (each trained one direction, tested the other, i.e. TSTR / TRTS):
- **Discriminator** — Isolation Forest & One-Class SVM trained to tell the two datasets apart;
  scored by **FPR** (a good/similar dataset fools the discriminator → high FPR → low dissimilarity).
- **Task** — Isolation Forest & One-Class SVM anomaly detectors trained on benign flows of one
  dataset, evaluated on the other; scored by **F1** (macro/micro/weighted).
- XGBoost task metric exists but is **commented out** (`test_xgboost_task_raw`), so `xgb_*` columns
  are always the `-1` error sentinel downstream.

Categorical/numerical attribute lists are defined (redundantly) in both `src/data_metrics.py` and
`src/data_metrics_framework.py` — keep them in sync if editing.

## Encoding (`src/binarize_queensland.py`)

The ML metrics operate on a fully binarized feature space (`encode_dataset` in
`data_metrics_framework.py`): IPs split per-octet then binarized, ports/flags binarized to bits,
protocol one-hot (TCP/UDP/ICMP/IGMP/OTHER), bytes/packets log-binned, duration linearly binned. The
result is cast to `int8`.

## Pipeline / how to run

Everything runs from the repo root as `python3 <script>_cmd.py --...`. The `*_cmd.py` scripts are
thin argparse wrappers around `src/`. Canonical command sequences live in **`examples/commands.txt`**
(runnable against the bundled `test_samples/`) and **`readme.md`**. Typical order:

1. **`create_samples_cmd.py`** `-l <big.csv> -s <out_dir/>` — draw a 10k-row random sample from a
   large dataset. (`src`-free, self-contained.)
2. **`syntax_chk_cmd.py`** `--load_path <csv> --ds_id <id> --store_path <dir>` — validate NetFlow
   syntax; writes cleaned data to `<dir>/<id>/checked_data/` and per-file error counts to
   `<dir>/<id>/error_counts/`. (Used for synthetic data before evaluation.) → `src/syntax_check.py`
3. **`syn_eval_cmd.py`** `--ds1 <csv> --ds2 <csv> --expid <id> --store_path <dir>` — the core
   evaluation. Runs attribute, correlation, discriminator, and task metrics; writes many raw CSVs
   to `<dir>/<id>/similarity_raw/<ds1>_<ds2>/`. → `src/data_metrics_framework.py`
4. **`syn_aggregate_cmd.py`** `--load_path <.../similarity_raw> --ds_id <id> --store_path <out.csv>`
   — reduce the raw per-pair CSVs to one aggregate row per dataset pair. → `src/aggregate_results.py`
5. **`syntax_chk_aggregate_cmd.py`** `--load_path <.../error_counts/> --store_path <out.csv>` —
   aggregate syntax error counts. → `src/syntax_check.py`
6. **`dissimilarity_table_cmd.py`** `--load_path <aggregate.csv> --syntax_path <syntax.csv>
   --store_path <out.csv>` — merge results + syntax, invert "higher = more similar" metrics, and
   compute the final **Data / Domain Dissimilarity Scores** as means.
7. **`plot_results_cmd.py`** `--prefix <dir/> --save_path_real ... --save_path_syn1/2 ...
   --path_syntax_syn1/2 ...` — box plots, metric-correlation heatmaps, and training-history line
   plots (PDF). → `src/plot_results.py`

Note: `src/aggregate_results.py` and the `aggregate_raw_results` function inside `src/plot_results.py`
are near-duplicates — the plotting module has its own copy plus history/heatmap plotting.

## CIDDS-001 support (`src/cidds.py`) — added for RQ2 of `../mp-lissmann`

Enabled by passing `--vocab_dir <dir with ip_vocab.json + port_vocab.json>` to
`syntax_chk_cmd.py` and `syn_eval_cmd.py`. Without the flag every code path is byte-for-byte
the original one, so the Queensland datasets are unaffected.

Why it is needed: `../mp-lissmann` trains TabDDPM on CIDDS flows whose IPs were bucketed onto
37 one-hot slots (35 named internal hosts + `External` + `Unknown internal`) and whose ports
were bucketed onto 19 (named ports + `<=1024` / `>1024`). 89.8 % of destination addresses and
93.5 % of source ports are therefore already gone before training. Two consequences:

- `check_netflows` rejects every such row as malformed (0/10,000 pass), which measures that
  repo's preprocessing rather than the generative model;
- `split_df_seperator` / `decimalToBinary` silently turn `External` and `>1024` into
  `0.0.0.0` and port 0 (the exception is swallowed), so the ML metrics run on fiction.

The CIDDS path instead: normalizes protocol *names* to protocol bytes (CIDDS exports carry
`TCP`/`UDP`, so this is needed for real CIDDS too — without it every real CIDDS row also fails
the syntax check and lands in `Protocol_OTHER`); one-hots the address/port tokens instead of
splitting them into octets and bits; and replaces the format checks with cross-field semantic
invariants (packets↔bytes agreement per direction, `bytes >= packets * 28`, non-empty flow,
UDP-without-TCP-flags). Those hold with **zero** exceptions in 300,000 real train rows, so
each violation is a genuine generation error.

**The comparison is only meaningful against a reference put through the same bucketing** —
`../mp-lissmann/prework/export_netflow_samples.py` writes it — and only alongside a
real-vs-real floor run. In the raw alphabet, two disjoint samples of the *same* real data
score mean JS 0.10 (L4_SRC_PORT alone: 0.68).

`--eval_ds` switches the task metric from TRTS to TSTR (see `data_metrics.split_task_frames`):
both detectors train on their own benign flows and are scored on one shared real,
attack-carrying sample. Required whenever both datasets are benign-only, as an unsupervised AD
generator's output is by design — the original setup then trains and grades on a single class
and F1 is meaningless.

## Scoring convention

Metrics where "higher = more similar" (discriminator FPR, task F1) are inverted (`1 - x`) in
`dissimilarity_table_cmd.py` / `plot_results.py` so that **higher final score = more dissimilar**.
A `-1` value anywhere is an **error sentinel** meaning a metric/file was missing; `remove_error_rows`
in `plot_results.py` drops such rows.

## Environment

- Python **3.9** (see `.venv/pyvenv.cfg`).
- Dependencies pinned in `requirements.txt`: numpy 1.25.1, pandas 2.0.3, scikit-learn 1.3.0,
  scipy 1.11.1, tqdm, seaborn 0.12.2, matplotlib 3.7.2. (`xgboost` is imported only in commented-out
  code and is **not** a dependency.)
- Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## Directory layout

- `src/` — library modules (imported by the `*_cmd.py` wrappers as `from src import ...`).
- `test_samples/` — small bundled real & synthetic 10k-row samples for smoke-testing the pipeline.
- `test_data/`, `data/` — generated pipeline outputs (`test_data` is gitignored).
- `experimental data/` — zipped source datasets and pre-computed `evaluated_data/` results + plots.
- `examples/commands.txt` — the authoritative, copy-pasteable end-to-end command sequence.

## Conventions & gotchas when editing

- The code is research-grade: heavy commented-out blocks, `print`-based debugging, broad
  `try/except` that swallows errors into `-1` sentinels, and `quit()` calls left in some functions
  (e.g. `calculate_ks_test` in `data_metrics.py` hard-`quit()`s — it is not usable as-is).
- `-1` sentinels must be dropped *before* the `1 - x` inversion in `dissimilarity_table_cmd.py`,
  which maps them to `2.0` — outside the [0,1] range of every real metric, so they dominate any
  mean they enter. The six `xgb_*` columns are always `-1` (xgboost is disabled and not a
  dependency) and two of them sit in `domain_metrics`; unfiltered they pinned the Domain
  Dissimilarity Score at ≈0.76 regardless of the data. `dissimilarity_table_cmd.py` now filters
  them; `src/plot_results.py` has its own copy of this logic (`remove_error_rows`, which drops
  whole rows instead) and has **not** been updated — check it before trusting the plots.
- Paths are built with string concatenation (e.g. `store_path+'/'+id+'/...'`); trailing slashes in
  `--store_path` matter. Follow the exact forms in `examples/commands.txt`.
- Directory names encode metadata and are parsed by `split('_')` — the pair dir is
  `<ds1name>_<ds1sample>_<ds2name>_<ds2sample>`; aggregation has a fallback split for the longer
  synthetic naming (`<name>_seed_<n>_step_<s>_...`). Renaming files/dirs will break parsing.
- Match the surrounding style (snake_case, no type checker, minimal abstraction) rather than
  refactoring broadly. Keep the duplicated attribute lists and the two `aggregate_raw_results`
  copies consistent if you touch either.
