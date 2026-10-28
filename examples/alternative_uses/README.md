# Example: fifteen alternative uses

A small, complete data folder that shows the instrument working without an API key.

| File | Contents |
|---|---|
| `ideas.csv` | 15 alternative uses of a tin can, a brick and a ping pong ball (researcher-written stimuli from Calic et al., 2020, *Scientific Reports* 10:2144) |
| `tasks.csv` | The three tasks |
| `ratings_long.csv` | Novelty and usefulness on a 1-5 scale from 39 raters. **The ratings are synthetic**: they were generated to resemble human ratings and no real rater's responses are included. Use them to see how the tools work, not to estimate anything. |
| `judge_scores.csv` | Scores from five AI judges under the stability grid P0 to P6 (`benchmark/prompts/perturbations_v1.1.yaml`), written by `benchmark.judge` |

```bash
python -m benchmark.agree        # the AI judge diagnostic
python -m benchmark.variance     # the variance pipeline
python -m benchmark.figures      # the diagnostic figure
```
