# Judge prompts

Every prompt the harness sends to a model lives here as a versioned YAML file. Rules:

1. **A released version is never edited.** To change anything in the wording, scale, anchors or output format, copy the file to a new version number (`pointwise_v1.1.yaml`), and describe the change in that file's `changelog`.
2. **Versions are `MAJOR.MINOR`.** A minor bump rewords or extends without changing what is measured. A major bump changes the scale, the dimensions or the output schema, so scores across major versions are not comparable.
3. **Every cached model call records the prompt id and version** (see `benchmark/judge.py`). A result can always be traced to the exact text that produced it.
4. **`status: draft`** means the prompt is not yet final. The judge prompt is rendered from the rubric file (`configs/rubrics/`), so use your own raters' verbatim instructions there.
5. **Perturbations are not prompts.** `perturbations_v1.2.yaml` describes transformations applied to a base prompt at run time by `benchmark/perturb.py`. The rendered text of every perturbed prompt is stored in the cache alongside the response.

Files:

| File | Purpose |
|---|---|
| `pointwise_v1.1.yaml` | One idea per call, every rubric dimension on the human scale, JSON output, rendered from the shared rubric. The default judge. |
| `pointwise_batch_v1.1.yaml` | Up to ten ideas per call (perturbation P4). |
| `perturbations_v1.2.yaml` | The stability grid P0 to P6, the default. The P1 rewordings were written by an LLM (qwen/qwen3.8-27b with `reword_v1.0`) and frozen; the file records the model and the call. |
| `perturbations_v1.1.yaml` | The same grid with hand-written P1 rewordings, used in the first pilot runs (the scores in `examples/alternative_uses/`). |
| `reword_v1.0.yaml` | Asks a model to reword the rubric statements without changing their meaning; `python -m benchmark.perturb reword` freezes the result into a new perturbation file (for example `perturbations_v1.2.yaml`) that records the generating model. |
| `expected_spread_v1.0.yaml` | The variance pipeline: the judge forecasts how ten raters would split across the scale. |
| `pairwise_v1.0.yaml` | Two ideas per call; retired before any run. |

The perspective and rater-twin sources of the variance pipeline reuse `pointwise_v1.1` with one added sentence (perspective) or a block of one rater's own past scores (rater twin); their exact text is built in `benchmark/variance.py` and stored with every cached call.
