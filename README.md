# TACC FlexServ Benchmark Notebook — Task 3

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nairr-portal/tacc-sandbox-3/blob/main/tapis_flexserv_benchmark_task_03.ipynb)

**[View the notebook](./tapis_flexserv_benchmark_task_03.ipynb)**

This notebook demonstrates training a random forest model to predict the **bulk modulus** (resistance to uniform compression) of inorganic crystalline compounds, using the UT Austin TACC Vista cluster, TACC's Tapis platform, and a FlexServ instance. It:

1. **Authenticates and initializes FlexServ** — connects to the TACC/Tapis platform, submits and monitors a FlexServ job, and loads a specified LLM.
2. **Prepares data** — embeds and writes `compound_elastic_properties_train.csv` and `compound_elastic_properties_test.csv` directly into the notebook for self-containment. Each row describes one crystal's formula and 3D atomic structure. The test file's true bulk modulus values are replaced with a dummy `0`, matching the official benchmark's data-contamination mitigation — the model has to actually predict, not read off the answer.
3. **Trains and predicts, with self-debug** — the LLM is given the task instruction, dataset preview, and a domain-knowledge hint (see below), and must turn the raw `formula`/`structure` columns into real numeric features before a model can use them, then train a random forest regressor and predict bulk modulus on the held-out test compounds. Its generated code is actually executed and its output checked; on a crash, a missing output file, a wrong output schema, or misaligned rows, the error is fed back to the model for another attempt, up to 10 retries (see below).
4. **Scores the result** — computes RMSE between the model's final predictions and the true (held-out) values, and reports pass/fail against the same 24.0 threshold the official ScienceAgentBench evaluator uses.

## Self-debug loop

Same technique as `tacc-sandbox-5`: starts from ScienceAgentBench's own `agent.py` self-debug baseline (`use_self_debug=True` — only checks for a crash or a missing output file) and extends it with an output-schema check the published technique doesn't do. Here that means: does `pred_results/compound_bulk_modulus.csv` have the required `material_id`/`K_VRH` columns, and does its `material_id` column match the test set's, in the same order — this task's real scoring cell compares predictions to gold by direct row-order equality (no merge), so row alignment is as much a formatting requirement as the column names.

This task's instruction already spells out the required columns explicitly ("material_id and K_VRH"), so the exact column-naming ambiguity that motivated the check on task 5 is less likely here — the check is included anyway for row-order/count coverage and to keep both notebooks' techniques consistent. Same boundary as task 5: the check only reads `material_id` values already visible to the model in the test CSV, never predicted `K_VRH` values or the withheld gold labels — no test-set feedback leaks through this loop. Also hardened the final scoring cell the same way task 5's was: it now prints a clean FAIL if the required columns are missing instead of raising a raw `KeyError`. Validated the loop mechanics against a mocked LLM before publishing (crash → wrong column name → right columns/wrong row order → correct, confirming all three failure modes get caught and fixed) — the test script itself is a scratch artifact, not part of this repo.

## Different from `tacc-sandbox-7` and `tacc-sandbox-66`: real numeric scoring, not eyeballing

The earlier two notebooks in this series (`tacc-sandbox-7`, `tacc-sandbox-66`) are both Info Visualization tasks — "success" there means a human (or, in the official benchmark, a GPT-4o visual judge) looks at two images and decides if they're similar enough. That's a soft, subjective bar.

This task is a Model Development task instead: the benchmark's own success criterion is a hard number (RMSE ≤ 24.0) computed against real, held-out ground truth. There's no picture at the end of this notebook — the final cell prints an actual RMSE and a PASS/FAIL verdict. This is a meaningfully stronger test of whether the LLM produced something that actually works, not just something that looks plausible.

I validated this before publishing: the benchmark's own reference solution (`matminer` composition/structure featurization → `RandomForestRegressor`) was run against the exact data embedded in this notebook and scores RMSE ≈ 22.45, which passes. So the 24.0 bar is a real, clearable target — if the LLM's one-shot attempt fails it, that's a genuine finding about the LLM, not an unreasonable threshold.

## Prompt variant: with domain knowledge — and why, this time

Unlike `tacc-sandbox-66` (which reverted to plain direct prompting after a knowledge hint was judged "too big a hint"), this notebook keeps a `domain_knowledge` block, for a different reason: an earlier direct-prompting attempt found a real **data leakage** shortcut. The dataset's `elastic_anisotropy`, `G_VRH`, and `poisson_ratio` columns are other elastic properties of the same compounds, related to the target `K_VRH` by elasticity theory (K = 2G(1+ν)/(3(1−2ν)) for isotropic materials) — a model can predict `K_VRH` very accurately from them without doing any real materials-science feature engineering, but such a model would be useless for the task's actual purpose (screening a hypothetical material that hasn't been characterized yet, and so wouldn't have those other properties available either).

The `domain_knowledge` block has two parts: (1) the official benchmark's own annotation for this task, pointing at Matminer for composition/structure featurization; (2) an explicit note ruling out the leakage columns as inputs. Unlike the CSV-parsing tip removed from `tacc-sandbox-66` (a software quirk nobody would know without hitting the bug), this is judged legitimate domain expertise — a materials scientist would know upfront that K, G, and ν aren't independent, the same way the task instruction's author clearly did (the reference solution explicitly drops those columns before training). It's stating a constraint of the assignment, not handing over the solution's code.

If a run fails or times out on unrelated implementation bugs (see the notebook history / project README for examples — hallucinated library import paths, wrong structure-serialization format assumptions), re-run the prompt/execution cells to sample again; `temperature=0.2` leaves enough randomness that repeated attempts aren't identical.

## A note on dependencies

Unlike the earlier two notebooks, this task realistically requires `scikit-learn` and `matminer` (which pulls in `pymatgen` and other materials-science tooling) for the LLM's generated code to have any chance of running at all — the official benchmark's evaluation harness auto-installs whatever packages a generated program imports, which this notebook doesn't replicate, so `scikit-learn` and `matminer` are pre-installed in the base setup cell instead. Installing `matminer` takes a little while the first time.

## Running locally

Always use a local `venv/` in this directory when running Jupyter here — do not use a system or other environment's Python. On macOS, prefer a Homebrew Python (`python3.12`) over the system `/usr/bin/python3`, which links against an outdated LibreSSL and triggers `urllib3` warnings on every HTTPS request this notebook makes.

```
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate
pip install jupyter notebook jupyterlab
jupyter lab   # or: jupyter notebook
```

The notebook's own runtime dependencies (`tapipy`, `pandas`, `matplotlib`, `scikit-learn`, `matminer`, etc.) are installed by its own first code cell — no separate requirements file needed. Featurizing the full dataset with `matminer` takes roughly 1–2 minutes.

## Source

Ported from ScienceAgentBench instance ID 3 (`Computational Chemistry` / `Feature Engineering, Machine Learning`), adapted from `hackingmaterials/matminer_examples`'s `machine_learning-nb/bulk_modulus.ipynb`.
