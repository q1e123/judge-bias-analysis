---
type: project
status: planned
tags: [project, research-plan, evaluation, llm-as-judge, bias, pentesting, benchmarking, reproducibility]
relates: [evaluating-pentesting-llms, llm-assisted-pentesting, CIPHER, adversarial-machine-learning]
target-venues: [international-journal-of-computers-and-applications, novel-approaches-for-cybersecurity-and-cyber-defense]
started: 2026-06-10
duration: ~8 weeks
---

# Project: Judge-LLM Bias in Pentest Evaluation

A 1–2 month empirical study measuring how much the **choice of LLM judge** distorts the evaluation of LLM penetration-testing assistants, and validating a drop-in mitigation. Fills documented gaps in [[evaluating-pentesting-llms]] (judge-LLM bias, manual-scoring bias) and directly stress-tests the FARR-Flow evaluation in [[CIPHER - Cybersecurity Intelligent Penetration-Testing Helper for Ethical Researcher]]. `[src: me]`

## 1. Motivation & positioning

LLM-as-judge is now the default way to score offensive-security LLMs (FARR Flow uses Llama-3-70B as judge; [[Generative AI for web application pentesting]] uses qualitative expert comparison). But self-preference / size / verbosity / position biases in LLM judges are documented **in general NLP** (Panickssery et al. 2024; Zheng et al. MT-Bench 2023). They are **not measured in the security domain**, where they would silently inflate or reorder published leaderboards. `[src: me]`

- **Novelty (defensible):** *first systematic measurement of judge-LLM bias in the pentest/security-evaluation setting, with a validated mitigation and a public reproducible artifact.* `[src: me]`
- **NOT the claim:** "we discovered self-preference bias." That is known — cite it, don't claim it. The contribution is domain-specific quantification + consequence (it changes who "wins") + a fix. `[src: me]`
- **Concrete hook:** [[evaluating-pentesting-llms]] already flags that Llama-3-70B-as-judge inflated its own FARR score and that Qwen-72B/GPT-3.5 looked less biased — this study turns that anecdote into a measured result. `[src: CIPHER]`

## 2. Research questions & hypotheses

| RQ | Question | Hypothesis |
|----|----------|------------|
| **RQ1** | Do judges score their own outputs higher than other judges score the same outputs? (self-preference) | H1: yes, statistically significant self-preference for most models. |
| **RQ2** | Independent of correctness, are larger or same-family models scored higher? (size / family bias) | H2: positive size bias; weaker family bias. |
| **RQ3** | How much does the choice of judge reorder the solver leaderboard? | H3: top-rank identity changes across judges; Kendall's τ between judges well below 1.0. |
| **RQ4** | Do mitigations (judge-panel median, identity-blinding, position-swapping) reduce bias without hurting agreement with human ground truth? | H4: panel + blinding cut self-preference materially and improve human agreement. |

## 3. Datasets — reuse existing open data (no bespoke task authoring)

**Design decision `[src: me]`:** the focus is judge *bias*, not pentest capability, so we do **not** build a FARR-Flow task set. We reuse existing open datasets. Two constraints drive the choice: (1) judge bias only manifests on **open-ended / free-text** responses (auto-gradable MCQ has no judge step, so no bias to measure); (2) for self-preference (RQ1) and ranking volatility (RQ3) we don't need reference answers at all — bias is measured *relative to the consensus of the other judges*. Ground-truth is needed only to separate bias from genuine quality and to rank judge trustworthiness (RQ2/RQ4), and an **existing human-labelled** dataset supplies it.

| Role | Dataset | What we use | Why | License |
|------|---------|-------------|-----|---------|
| **Primary** (security, open-ended, judge-required) | **CyberSecEval** (Meta PurpleLlama) | MITRE (cyberattack-compliance), Spear-Phishing, Prompt-Injection, Code-Interpreter-Abuse categories | Free-text prompts **already scored by a swappable LLM judge** — we reuse their prompts *and* judge harness and plug in our judge pool. Eliminates all task authoring. | MIT |
| **Human anchor** (ground truth) | **RewardBench** (AllenAI) | Safety subset (refusals-dangerous/offensive, xstest, do-not-answer) + Code/Reasoning subset | Human-curated chosen/rejected pairs → measure each judge's accuracy vs. ground truth **without hand-labelling**. | ODC-BY |
| **Cross-domain control** (general, non-security) | **MT-Bench human judgments** (LMSYS) | 80 open-ended Qs, 3.3K expert pairwise human prefs (6 models) | Quantify how bias differs *security vs. general*; reuses an existing human-vs-LLM-judge agreement baseline (>80%). | verify on card |
| **Optional probe** (objective-answer) | **CTIBench** (AI4Sec, NeurIPS'24) | CTI-RCM / CTI-VSP / CTI-ATE (structured, auto-checkable) | "Does self-preference persist even when correctness is objectively knowable?" Demoted because mostly auto-gradable, not open-ended. | on HF card |

Sources (verified 2026-06-10): CyberSecEval — [github.com/meta-llama/PurpleLlama](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks) (MIT; the MITRE/phishing/prompt-injection/interpreter-abuse tests are LLM-judge-graded, secure-code/canary-exploit/AutoPatch are auto-graded); RewardBench — [huggingface.co/datasets/allenai/reward-bench](https://huggingface.co/datasets/allenai/reward-bench) (ODC-BY); MT-Bench judgments — [huggingface.co/datasets/lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments); CTIBench — [huggingface.co/datasets/AI4Sec/cti-bench](https://huggingface.co/datasets/AI4Sec/cti-bench).

**Optional top-up hand-labelling (~30–50 items).** If the RewardBench security subset proves too small or off-distribution from CyberSecEval's offensive prompts, hand-label a small slice of CyberSecEval *solver outputs* against its rubric. This is now a fallback, **not** the load-bearing FARR build it was before. `[src: me]`

**Provenance / leakage note (still applies).** CyberSecEval and MT-Bench are public and likely in training corpora. Record dataset release date vs. each model's cutoff; where possible report a "likely-seen" split so memorisation isn't mistaken for quality. Bias measured *between judges on identical pooled answers* is largely robust to this, but flag it. `[src: me]`

> All datasets are downloaded into the project workspace under their own licenses, **not** copied into `raw/`. The released reproducibility artifact ships our **harness + judge prompts + all judgment records**, plus dataset *loaders/IDs* (not redistributed corpora, unless the license permits) — strong fit for [[international-journal-of-computers-and-applications]]. Honour MIT / ODC-BY attribution. `[src: me]`

## 4. Models

**Solver pool (generate answers) — and the same models serve as the Judge pool (score answers), enabling the N×N round-robin.** Target ~6–8 models spanning families and sizes:

| Model | Access | Role | Why included |
|-------|--------|------|--------------|
| GPT-4o (or current GPT-4-class) | Paid API | solver+judge | frontier baseline |
| Claude (Sonnet-class) | Paid API | solver+judge | cross-family frontier |
| Gemini (Flash/Pro) | Paid API | solver+judge | third family |
| Llama-3-70B | Groq | solver+judge | the FARR-Flow judge — central to the hook |
| Qwen-72B | Groq | solver+judge | flagged as less-biased in [[CIPHER...]] |
| Mixtral-8x7B | Groq | solver+judge | MoE mid-size |
| Llama-3-8B / Qwen-7B | Groq or local GPU | solver+judge | small-model size-bias anchor |
| (optional) CIPHER-style 7B | local GPU | solver | domain-finetuned point of comparison |

Design so families (OpenAI / Anthropic / Google / Meta / Alibaba / Mistral) and sizes (7B → frontier) are both spanned — required to separate **family bias** from **size bias** in RQ2. `[src: me]`

## 5. Experimental design

1. **Answer generation.** Every solver answers every primary-set prompt once (fixed decoding params, temp logged), reusing CyberSecEval's prompts directly. Store raw outputs + token counts. (For RewardBench/MT-Bench we can reuse the *existing* model responses where provided, or regenerate with our solver pool.)
2. **Round-robin judging (the core).** Every judge scores every solver's answer on every item:
   - **Pointwise:** rubric score 0–3 (outcome/service/vuln), as in FARR Flow.
   - **Pairwise:** for a sampled set of answer pairs, "which is better?" — pairwise exposes preference bias more cleanly than pointwise and feeds Bradley-Terry/Elo.
3. **Controls (run as conditions, not afterthoughts):**
   - **Position-swap:** every pairwise comparison run in both orders (A,B) and (B,A) → position-bias rate.
   - **Verbosity logging:** answer length recorded → correlate length with score.
   - **Identity-blinding:** a condition where the judge prompt strips any model self-identification / style tells.
4. **Mitigation conditions (RQ4):** (a) judge-panel = median/majority over ≥3 judges with self-scoring excluded; (b) blinded; (c) position-swapped. Compare each against the naive single-judge baseline.

**Scale sanity:** ~8 solvers × ~8 judges × ~200 items ≈ 12.8k pointwise judgments per condition. Cheap and fast on Groq; budget paid-API judges deliberately. Cache everything; make runs resumable.

## 6. Metrics (define precisely in the paper)

- **Self-Preference Index (SPI)** for model *m*: mean( score given by *m* to *m*'s own answers ) − mean( score given by all **other** judges to *m*'s answers ). Report with bootstrap 95% CIs. (RQ1)
- **Bias matrix** B[j,s] = mean score judge *j* gives solver *s*, row-normalised to expose inflation — **the headline heatmap.** (RQ1/RQ2)
- **Size-bias regression:** score ~ log(params) + correctness(anchor) + verbosity + judge FE; the params coefficient after controlling for correctness = size bias. (RQ2)
- **Family-bias term:** same-family(judge,solver) indicator in the regression. (RQ2)
- **Ranking volatility:** per-judge Bradley-Terry/Elo leaderboards; pairwise **Kendall's τ** and top-1 churn across judges. (RQ3)
- **Human agreement:** each judge vs. calibration labels — Cohen's κ / Spearman — identifies *trustworthy*, not merely *unbiased*, judges. (cross-cutting)
- **Position-bias rate:** % of pairs whose winner flips on order swap. (control)
- **Mitigation deltas:** Δ SPI and Δ human-agreement, naive vs. each mitigation. (RQ4)

## 7. Statistical analysis

- Bootstrap CIs for SPI and all means; paired tests (Wilcoxon) for own-vs-other judgments per model.
- Mixed-effects / OLS with judge and item fixed effects for the size/family regression.
- Inter-judge reliability: Krippendorff's α across the judge panel.
- Multiple-comparison correction (Holm) across the per-model SPI tests.

## 8. What has to be coded (harness modules)

```
judge-bias/
  config.py          # model registry: name, provider, endpoint, params, family, size
  clients/           # thin wrappers: groq, ollama, openwebui/openai_compat (all open-weight)
  dataset/
    loaders.py       # pull CyberSecEval / RewardBench / MT-Bench / CTIBench -> common schema
    schema.py        # TaskItem, Response, (optional) ReferenceLabel, Rubric; leakage tags
  generate.py        # solver answers (resumable, cached, token+param logging)
  judge.py           # pointwise + pairwise; position-swap; blinding; prompt templates
  aggregate.py       # SPI, bias matrix, Bradley-Terry/Elo, Kendall tau, kappa
  mitigate.py        # panel median / majority, blinded, swapped conditions
  stats.py           # bootstrap CIs, mixed-effects regression, Holm correction
  figures.py         # bias heatmap, SPI forest plot, ranking-flip diagram
  cache/  results/   # JSONL of every judgment (reproducibility artifact)
```

Engineering notes: one canonical JSONL judgment record (item_id, solver, judge, condition, raw_score, parsed_score, order, latency, tokens); fully resumable (key on the tuple, skip if cached); deterministic seeds/temps logged; rubric + all prompt templates version-controlled and shipped with the paper. `[src: me]`

## 9. Timeline (8 weeks)

Reusing open data frees ~1.5 weeks previously spent on task authoring/hand-labelling, reinvested into the cross-domain control + mitigation experiments.

| Wk | Milestone |
|----|-----------|
| 1 | Scope frozen; `loaders.py` + schema; CyberSecEval/RewardBench/MT-Bench pulled, licenses checked, leakage tags assigned |
| 2 | Clients + `generate.py`; all solver answers generated & cached on the primary (CyberSecEval) set |
| 3 | `judge.py` round-robin (pointwise+pairwise) across the full matrix on primary set |
| 4 | Position-swap & blinded conditions; replicate pipeline on MT-Bench (cross-domain control) |
| 5 | `aggregate.py`+`stats.py`+`figures.py`: SPI, bias matrix, rankings, regressions; judge-vs-human accuracy via RewardBench anchor |
| 6 | Mitigation experiments (RQ4: panel/blinding/swap) |
| 7 | Optional CTIBench objective-answer probe + top-up hand-labelling if needed; robustness checks |
| 8 | Write-up; release artifact; internal review |

## 10. Risks & mitigations

- **"This is already known" desk-reject** → lead with domain-specific consequence + mitigation + artifact; cite Panickssery/Zheng up front. (§1) `[src: me]`
- **Memorisation confound** → public datasets are likely in training data; rely on between-judge comparison on identical answers (robust to this) + leakage tags (§3).
- **Bias vs. quality conflation** → use RewardBench's existing human labels as the anchor; top-up hand-labelling only as fallback (§3).
- **Refusal confound (now central)** → CyberSecEval's offensive categories literally score *whether a model refuses/complies*, and refusal propensity is family- and size-dependent (cf. size-dependent "benign" behaviour in [[CIPHER...]]). A judge may conflate "safely refused" with "low quality." Mitigate by: separating a refusal-detection pass from the quality score, and reporting bias on the *non-refused* subset as well as overall.
- **Judge output unparseable** → structured-output prompts, retries, refusal logged as an explicit result class.
- **Cost creep on paid APIs** → Groq for the bulk matrix; cap paid-judge conditions; everything cached.
- **Small item count → weak power** → power check on the calibration subset early; expand items if CIs too wide.

## 11. Ethics & scope

Evaluation research over **published, licensed benchmark prompts** (CyberSecEval/RewardBench/MT-Bench) — no live targets, no novel weaponised exploits generated. Released artifact contains the harness, judge prompts, and scores, not redistributed offensive corpora beyond each dataset's license. Aligns with the dual-use cautions in [[risks-and-ethics-of-ai-in-offensive-security]]. `[src: me]`

## 12. Deliverables

- Paper (target [[international-journal-of-computers-and-applications]]; backup [[novel-approaches-for-cybersecurity-and-cyber-defense]] SI, deadline 30 Jan 2027).
- Public artifact: harness + judge prompts + dataset loaders + all judgment JSONL (datasets referenced by ID under their own licenses).
- Headline figures: bias matrix heatmap, SPI forest plot, judge-dependent ranking-flip diagram.

See also: [[evaluating-pentesting-llms]] · [[llm-assisted-pentesting]] · [[CIPHER - Cybersecurity Intelligent Penetration-Testing Helper for Ethical Researcher]] · [[adversarial-machine-learning]]
