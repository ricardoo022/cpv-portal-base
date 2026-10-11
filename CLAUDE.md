# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Model that suggests the CPV code (EU public procurement vocabulary, ~9 500 codes in a tree: division 2 digits → group 3 → class 4 → category 5 → full 8) from a Portuguese public contract description, trained on Portal BASE contracts (dados.gov.pt). The goal is a ranked top-5 of specific codes; division-level accuracy is only for comparison with published benchmarks. Plan: TF-IDF + logistic regression baseline, then fine-tune **BERTimbau base** and **Albertina PT-PT 100M** as classifiers on a rented GPU. AMALIA and Qwen were explicitly dropped.

All docs, notebooks, comments and replies are in **European Portuguese**. The user is learning ML through this project: go step by step, define every term with a concrete example, give one recommendation at a time, say which lessons generalize beyond CPV, and never present unverified facts as certain. Do not write day-by-day plans ("Dia 1, Dia 2").

## Commands

```
make help                                              # shortcuts for the commands below (Makefile)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.10
.venv/bin/python scripts/download_data.py [ANOS...]    # .xlsx from dados.gov.pt -> data/raw/
.venv/bin/python src/convert_to_parquet.py [ANOS...]   # .xlsx -> data/processed/raw/*.parquet (~1 min/file)
.venv/bin/python src/prepare.py                        # -> data/processed/{treino,validacao,teste}.parquet (~1 min)
.venv/bin/pytest                                       # all tests (~1 s)
.venv/bin/pytest tests/unit/test_prepare.py::test_build_train_empate_fica_o_menor_codigo   # one test
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploracao.ipynb  # run from repo root
```

Notebooks add `../src` to `sys.path`, so they must run with `notebooks/` as the working directory (nbconvert and the editor do this).

## Git and CI

- Work on a branch and open a PR to `main` with `gh pr create`; don't commit straight to `main`. Commit messages and PR descriptions are in Portuguese.
- `.github/workflows/testes.yml` runs the whole `pytest` suite on every push (any branch) and every PR, in about 40 s. Tests use only hand-made data, so CI never needs `data/`.
- The repo is private on the free plan, so `main` has no branch protection: wait until the CI runs for the PR's head commit are green before `gh pr merge --merge --delete-branch`.
- The installed `gh` is 2.4.0: `gh run list` has no `--branch` flag and no `workflowName` JSON field. Filter runs with `--json headSha,status,conclusion` against `git rev-parse HEAD`.
- GitHub's "Dependency Graph" runs on `main` are automatic (it reads `requirements.txt`), not ours.

## Architecture and data flow

`data/raw/*.xlsx` → `convert_to_parquet.py` (streams rows with openpyxl, writes **every column as text**, no transformation) → `data/processed/raw/contratosAAAA.parquet` → `prepare.py` → three output files. `data/` and `.venv/` are not in git.

- `src/cpv_data.py` holds the rules shared by the pipeline and the notebooks: CPV code extraction, code level (trailing-zeros heuristic), text normalization, and the split constants `TRAIN_START` (2022-01-01), `VAL_START` (2025-10-01), `TEST_START` (2025-11-01), `YEAR_END` (2026-01-01). Change split dates only here.
- **The file year (and the `Ano` column) is the contract's signing year, not its publication year.** Contracts published in a given year are spread across several files, so always read all files and filter by `dataPublicacao` (`load_contracts` does this per file to save memory).
- `src/prepare.py` is a sequence of small testable functions (`load_one` → `dedupe_ids` → `add_columns` → `build_train` / `build_eval` → `check_sets` → save), orchestrated by `run(raw_dir, out_dir)`. Each step maps to a numbered decision in `docs/qualidade_dados.md`. Key rules:
  - input text = `objectoContrato` + `descContrato` (only if different) + `DescrAcordoQuadro` (if present), joined with `" | "`;
  - label = first CPV code; all codes kept in `codigos` (space-separated) for evaluation;
  - **train only**: drop division-level codes (`XX000000`), then one example per normalized text with its most common code (tie → smallest code); `n_contratos` keeps the original count;
  - validation/test keep every contract and get `visto_no_treino` (text also present in train), to measure memorization separately;
  - text identity uses a 64-bit hash of the normalized text (memory).
  - `check_sets` raises `DataCheckError` (not `assert`, which `-O` would disable) on leakage between sets, dates outside a period, division-level codes in train, repeated train texts, malformed codes, or "None"/"nan" written into text.
- `src/labels.py` (`collapse_rare_codes`) builds the training label at training time, not in `prepare.py`: a code with fewer than N training examples backs off to its class (4 digits), then its group (3 digits), else the example is dropped; never to the division. N is a hyperparameter (5, 10 or 20, default 10) to be chosen with TF-IDF on validation; counts come from train only. Evaluation keeps the original label.
- `src/evaluation.py` scores ranked suggestions (pandas Series of 5-code lists) against `codigos`; it reads and writes no files. Measures: `codigo` (exact match; only contracts with a code below division level count, since the model never suggests divisions), `classe` (4 digits, codes of level >= 4), `divisao` (2 digits), plus a `so_divisao` row for contracts whose codes are all division-level. `report` splits all vs. new text (`visto_no_treino == False`); `hierarchical*` give partial credit (Kiritchenko hP/hR/hF) for display only.
- `src/baselines.py` is the "most common codes" baseline (E4-T2): always suggest the 5 most frequent training labels, counted once per training row after `collapse_rare_codes` with the model's N (tie → smallest code). Weighting rows by `n_contratos` is optional and only for comparison: it gave an easier baseline on validation (decision 10). Its validation numbers (`notebooks/03_regua.ipynb`, `make regua`) are section 12 of `docs/qualidade_dados.md`; the same 5 labels come out for N = 5, 10 and 20.
- `src/comparison.py` decides between two options with a paired bootstrap (1 000 samples, same sample for both, a difference counts if one side wins in >= 95%). `choose` applies the E4 rule: top-5 on `codigo` decides, top-5 on `classe` is the guard, ties fall back to a default. `compare` runs it for every measure × metric (~40 s on 24k contracts).
- Planned for phase 4 (`docs/workflow.md`): a quality gate. Real training (laptop or rented GPU) writes validation metrics to a small committed file (e.g. `metrics/validacao.json`); a CI test reads it and fails below a threshold. Thresholds are measured on validation, never on test, and are relative (TF-IDF must beat always-predict-the-most-common-code; BERTimbau/Albertina must beat TF-IDF), not invented numbers. CI also runs a tiny training smoke test on hand-made data.
- The laptop has ~5.8 GB RAM shared with the editor; `prepare.py` peaks around 2.7 GB. Keep processing per file and avoid extra copies of the text columns.

## Conventions

- `src/` holds only pipeline code (code that transforms data, trains or evaluates). Exploration, counts and one-off analysis go in `notebooks/`; rules that must always hold become checks in the pipeline and tests.
- Every number written in `docs/*.md` must come from a notebook cell or pipeline output, not from ad-hoc terminal commands. Sections 1–6 of `docs/qualidade_dados.md` were measured on the 2025/2026 files only; sections 7–10 (`notebooks/01_exploracao.ipynb`) use the final train (Jan 2022–Sep 2025, all files), and section 10 reads `data/processed/treino.parquet`, so `prepare.py` must run before that notebook. Its train-contract count and "outside the most common code" count must match the `prepare.py` funnel (section 11).
- Never look at the test set (contracts published Nov–Dec 2025) during exploration or decisions; use the validation set (Oct 2025) to compare options. 2026 data is held in reserve.
- `docs/workflow.md` (CRISP-DM phases with checkboxes) is the plan and status; `docs/backlog.md` holds the epics, user stories, technical tasks and spikes with acceptance criteria (done epics and the next one, written with the `writing-epics-and-stories` skill); `docs/qualidade_dados.md` holds measurements and numbered decisions; `docs/dados.md` describes the 39 columns and whether each exists when the CPV is chosen (columns known only after award are leakage); `docs/prs/pr_<ID>.md` holds each PR description (committed, written with the `writing-pull-requests` skill).
- When changing a rule in `prepare.py`, update the hand-made contracts in `tests/integration/test_pipeline.py`, which builds `.xlsx` files and runs the full pipeline end to end. New tests should be checked by injecting the bug they target (mutation testing).
- The user often has notebooks open in Cursor, which can save a stale version over a regenerated one. Check the file on disk (cell count, mtime) before and after rewriting a notebook, and don't commit a notebook that changed unexpectedly.
