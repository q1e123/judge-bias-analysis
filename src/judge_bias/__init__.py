"""judge_bias - harness for measuring judge-LLM bias in security evaluation.

See ``judge-llm-bias-study.md`` for the research design. Module map:

- ``config``        model registry (name, provider, family, size, role)
- ``clients``       thin provider wrappers
- ``dataset``       common schema + dataset loaders
- ``prompts``       pointwise / pairwise judge prompt templates
- ``generate``      solver answer generation (resumable, cached)
- ``judge``         round-robin pointwise + pairwise judging
- ``aggregate``     SPI, bias matrix, Bradley-Terry/Elo, Kendall tau, kappa
- ``mitigate``      judge-panel / blinding / position-swap conditions
- ``stats``         bootstrap CIs, mixed-effects regression, Holm, Krippendorff
- ``figures``       bias heatmap, SPI forest plot, ranking-flip diagram
"""

__version__ = "0.1.0"
