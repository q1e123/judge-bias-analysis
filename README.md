# judge-bias-analysis

Harness for **measuring judge-LLM bias in security evaluation** - the empirical
study behind *Who Judges the Judges? Measuring Self-Preference and Family Bias
in LLM-as-Judge Evaluation of Cybersecurity Tasks*.

It runs an NxN round-robin where every model both *answers* security prompts
(solver) and *scores* every model's answers (judge), then quantifies
self-preference, family bias, ranking volatility, position bias, and the effect
of two drop-in mitigations (identity blinding, panel-median) against an
objective, auto-checkable ground-truth anchor.

**All models are open-weight.** The released study serves the full pool through
**OpenRouter** (one OpenAI-compatible gateway, temperature 0, fixed seed), so a
complete run costs a few dollars and is fully reproducible. The client layer
also supports **Groq**, **Ollama**, and any OpenAI-compatible endpoint, so you
can point the pool at other hosts.

## Install (uv)

```bash
uv sync --extra dev              # base + test deps
uv sync --extra providers        # + groq SDK + openai SDK (HTTP client for OpenRouter/OpenWebUI/vLLM)
uv sync --extra data             # + HuggingFace dataset loaders
```

For real runs, copy `.env.example` to `.env` and set `GROQ_API_KEY`. In the
released study that variable holds the **OpenRouter** key (`sk-or-...`); the
registry points every model at `https://openrouter.ai/api/v1` via the
`openai_compat` provider.

## Reproduce the study

The `runs/` folder drives the whole pipeline in order, all reading one shared
`workspace_run/` and all resumable. Shared settings (pool, workspace, frozen
inputs, concurrency) live in `runs/_config.ps1` / `runs/_config.sh`.

```powershell
.\runs\00_check.ps1            # ping every model, confirm the key
.\runs\00_freeze_inputs.ps1    # sample + content-hash the item set (byte-identical across stages)
.\runs\01_generate.ps1         # solver answers
.\runs\02_judge.ps1            # round-robin pointwise + pairwise (76,800 judgments)
.\runs\03_anchor.ps1           # judge-vs-truth accuracy on WMDP pairs
.\runs\04_analyze.ps1          # all metrics -> results/analysis.json  (no model calls)
.\runs\05_report.ps1           # per-dataset table
```

The pool is `pool8_openrouter.json` (8 models, five families, 4B-120B). Only
`04_analyze` and `05_report` re-run cheaply once the judgment store exists.

## CLI (each stage is resumable)

```bash
uv run judge-bias models                                  # show the model registry
uv run judge-bias gen     --registry pool8_openrouter.json --dataset cyberseceval --n 35
uv run judge-bias judge   --registry pool8_openrouter.json --concurrency openai_compat=8
uv run judge-bias anchor  --registry pool8_openrouter.json --source wmdp
uv run judge-bias analyze --registry pool8_openrouter.json   # SPI/CIs, FE regressions, power, mitigations
uv run judge-bias report                                     # per-dataset table (security datasets only)
uv run judge-bias figs                                       # headline figures
uv run pytest -q                                             # 15 tests: metrics recover a known injected bias
```

Outputs land in the workspace: `results/{responses,judgments,anchor}.jsonl`,
`results/analysis.json`, `results/per_dataset.{csv,json}`, and `figures/*.png`.

## Module map

| File | Role |
|------|------|
| `config.py` | model registry (name, provider, family, size, role); `RunConfig`; `load_registry` |
| `clients/` | thin provider wrappers (`providers.py`: Groq, Ollama, OpenAI-compatible) + `get_client` factory |
| `dataset/schema.py` | `TaskItem`, `Response`, `Judgment`, `Rubric`, `ReferenceLabel`, leakage tags |
| `dataset/loaders.py` | CyberSecEval / MT-Bench / CTIBench / WMDP + offline synthetic |
| `cache.py` | append-only JSONL stores with tuple-keyed resume |
| `prompts.py` | versioned pointwise (0-3 rubric) + pairwise judge templates |
| `generate.py` | solver answers (resumable, cached, token/param logging) |
| `judge.py` | round-robin pointwise + pairwise; position-swap; blinding; objective anchor |
| `aggregate.py` | SPI, bias matrix, Bradley-Terry, Kendall tau, leniency, position bias, agreement |
| `mitigate.py` | panel-median / blinded conditions; mitigation deltas |
| `stats.py` | bootstrap CIs, paired Wilcoxon, the fixed-effects estimators, Holm, Krippendorff alpha |
| `figures.py` | bias heatmap, SPI forest plot, ranking-flip diagram |
| `provenance.py` | reproducibility manifest (model ids, dataset hashes, lib versions) |

## Metrics and research questions

- **Self-preference.** `aggregate.self_preference_index` / `stats.spi_with_ci`
  give the raw SPI (own score minus leave-one-out panel consensus) with a
  bootstrap CI, Holm-corrected Wilcoxon, Cohen's *dz*, and win-rate. Because raw
  SPI is confounded by a judge's leniency, the reported estimates come from
  fixed-effects models: `stats.self_preference_regression` (pooled `is_self` net
  of item + judge + **solver** fixed effects) and `stats.self_preference_by_judge`
  (per-judge `tau_j`). The per-judge adjustment is the headline result - it shows
  self-preference is strongly judge-specific and that the raw SPI ranking is
  mostly leniency.
- **Family and size bias.** `stats.family_bias_regression` estimates same-family
  bias with solver fixed effects (identified net of answer quality; significant).
  `stats.size_bias_consensus_deviation` measures size bias as a judge's deviation
  from consensus and finds **none** - the raw score-on-size gradient
  (`stats.size_family_regression`) is answer quality, not bias.
- **Ranking volatility.** `aggregate.bradley_terry_by_judge` + `ranking_volatility`
  (per-judge leaderboards, Kendall tau, top-1 churn).
- **Mitigations.** `mitigate.mitigation_deltas`: name-blinding shows no effect;
  a panel-median aggregate removes self-preference by construction.
- Controls: `aggregate.position_bias_rate`, `verbosity_correlation`,
  `judge_leniency`; reliability: `stats.krippendorff_alpha`.

### Robustness and validity

- **Prompt-robustness sweep** - `judge.judge_robustness` re-runs the pointwise
  matrix under paraphrased prompts and a rescaled rubric; `aggregate.robustness_summary`
  checks self-preference is stable. Run with `judge-bias robust`.
- **Pointwise-pairwise agreement** - `aggregate.convergent_validity`: Spearman
  between the pointwise and Bradley-Terry rankings.
- **Power / MDE** - `stats.power_analysis`: per-model sample size and minimum
  detectable effect, so nulls are not over-read.
- **Refusal confound** - `aggregate.refusal_breakdown` + `exclude_refused`: every
  estimator can be re-run on the non-refused subset.
- **Security-vs-general contrast** - `aggregate.domain_contrast` (with MT-Bench).
- **Reproducibility manifest** - `provenance.write_manifest`: model ids, dataset
  SHA-256 + leakage census, library versions, seed, written every run.

## Pointing the pool at other backends

The client factory (`clients/get_client`) routes by a spec's `provider`: `groq`
uses the Groq SDK; `ollama` talks to the native `/api/chat` endpoint over stdlib
HTTP; `openai_compat` / `openwebui` / `local` / `vllm` reuse the `openai` SDK as
an HTTP client against a custom `base_url`. Per-model `base_url` and
`api_key_env` fields point at custom hosts or alternate keys. Pass your own
`--registry models.json` (a JSON list of `ModelSpec` dicts) to swap the pool.
Requests run with **per-backend concurrency**: network backends parallelise up
to a cap, while a single-GPU Ollama runs serial and model-major so weights stay
resident instead of thrashing VRAM.

## Using real datasets

```bash
uv sync --extra data                 # HuggingFace dataset loaders
uv run judge-bias datasets           # list the catalog (role, HF id, license)
uv run judge-bias gen    --dataset cyberseceval --registry pool8_openrouter.json
uv run judge-bias anchor --source wmdp          --registry pool8_openrouter.json
```

Deliberately minimal: one dataset per role, all open, no human labeling. Four
roles - **primary** (security prompts models answer and judges score),
**anchor** (objective ground truth for judge accuracy), **control**
(general-domain, for the security-vs-general contrast), **probe** (auto-checkable):

| Dataset | Role | HF id | License | Notes |
|---------|------|-------|---------|-------|
| CyberSecEval | primary | `meta-llama/PurpleLlama` (local clone) | MIT | MITRE / prompt-injection / interpreter abuse - the core |
| WMDP | anchor / probe | `cais/wmdp` | MIT | correct-vs-wrong cyber pairs; judge accuracy = ranks the correct answer higher (no human labeling) |
| MT-Bench | control | `lmsys/mt_bench_human_judgments` | (card) | general-domain baseline for the domain contrast |
| CTIBench | probe | `AI4Sec/cti-bench` | (card) | structured CTI tasks (auto-checkable) |

The bias metrics need only CyberSecEval; MT-Bench adds the domain contrast; WMDP
is the on-domain, label-free ground-truth anchor (a judge is "trustworthy" if it
ranks the objectively-correct security answer above a wrong one). CyberSecEval
needs a local PurpleLlama clone. Every item carries a leakage tag, and the
provenance manifest writes a leakage census each run.

## Tests

`tests/test_metrics.py` builds a small judgment set with a *known* structure - a
size effect, a same-family effect, and per-judge self-preference - directly, with
no model calls, and asserts the analysis layer recovers those signs. So a green
`pytest` (15 tests) means the estimators are wired correctly before a single API
call is spent.

## Ethics and scope

Evaluation research over published, licensed benchmark prompts - no live targets,
no novel weaponised exploits. Raw solver answers for the offensive categories are
withheld from the default release and shared with vetted researchers on request.
